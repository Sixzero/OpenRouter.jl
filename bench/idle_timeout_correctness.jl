#!/usr/bin/env julia
# End-to-end correctness audit of the idle-timeout through streamed_request!.
# (a) stalled stream -> StreamIdleTimeoutError
# (b) healthy slow stream (chunks every ~0.3s) NOT killed by tighter timeout
# (c) disabled (timeout=0) behaves like plain readavailable
# Plus: isolated abort-detection timing (server sleeps inflate wall time because
# HTTP cleanup waits on the handler).
using OpenRouter
using OpenRouter: StreamIdleTimeoutError, readavailable_with_idle_timeout, _abort_read!,
                  HttpStreamCallback, ChatCompletionSchema, streamed_request!
using HTTP, Sockets, Printf, Test

function serve_chunks(chunk_fn; ) # chunk_fn(http) writes the body
    server = HTTP.serve!("127.0.0.1", 0; stream=true, listenany=true) do http
        # Drain the request body the client POSTs, else the handler can race
        # ahead and the conn EOFs on the client side.
        while !eof(http); readavailable(http); end
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        chunk_fn(http)
    end
    port = HTTP.Sockets.getsockname(server.listener.server)[2]
    (server, "http://127.0.0.1:$port")
end

sse(data) = "data: $(data)\n\n"
chatchunk(txt) = sse("""{"choices":[{"delta":{"content":"$txt"},"index":0}]}""")
donechunk() = sse("[DONE]")

function run_stream(url; idle_timeout)
    cb = HttpStreamCallback(out=IOBuffer(), schema=ChatCompletionSchema(),
                            kwargs=(; stream_idle_timeout=idle_timeout,
                                      reuse_limit=0, retry=false))
    streamed_request!(cb, url, ["Content-Type"=>"application/json"], "{}")
    cb
end

# Warm up the full streaming path once (JIT) so timing asserts measure the
# mechanism, not first-call precompilation (~2s).
let
    srv,url = serve_chunks() do http
        write(http, chatchunk("warm")); flush(http)
        write(http, donechunk()); flush(http)
    end
    run_stream(url; idle_timeout=0.0)
    close(srv)
end

@testset "Idle-timeout end-to-end via streamed_request!" begin

    @testset "(a) stalled stream -> StreamIdleTimeoutError" begin
        # send 2 chunks, then block forever WITHOUT closing the socket.
        # NOTE: the handler must keep the socket alive but NOT sleep on the
        # request task — a handler sleep() makes HTTP *client* teardown wait on
        # it, inflating end-to-end wall time (the documented gotcha). We park on
        # a Condition that never fires; the socket stays open, handler task
        # idle, so the only thing the client waits on is the idle timeout.
        done = Ref(false)
        srv,url = serve_chunks() do http
            write(http, chatchunk("hi")); flush(http)
            write(http, chatchunk("there")); flush(http)
            # park without a handler sleep() inflating client teardown; bail out
            # cleanly once the client aborted the connection.
            try
                while !done[] && isopen(http); sleep(0.05); end
            catch; end
        end
        t0 = time()
        err = nothing
        try
            run_stream(url; idle_timeout=0.5)
        catch e
            err = e
        end
        dt = time() - t0
        done[] = true
        @test err isa StreamIdleTimeoutError
        @test err.timeout == 0.5
        @printf("    abort detected in %.3fs (timeout=0.5s, handler parked not sleeping)\n", dt)
        @test dt < 3.0   # detected promptly
        close(srv)
    end

    @testset "(b) healthy slow stream NOT killed" begin
        # chunk every 0.3s for ~1.5s total; timeout 0.5s must NOT fire
        srv,url = serve_chunks() do http
            for i in 1:5
                write(http, chatchunk("tok$i")); flush(http)
                sleep(0.3)
            end
            write(http, donechunk()); flush(http)
        end
        cb = run_stream(url; idle_timeout=0.5)
        @test length(cb.chunks) >= 5   # all content chunks survived
        close(srv)
    end

    @testset "(c) disabled (timeout=0) == plain readavailable" begin
        srv,url = serve_chunks() do http
            write(http, chatchunk("a")); flush(http)
            write(http, chatchunk("b")); flush(http)
            write(http, donechunk()); flush(http)
        end
        cb0 = run_stream(url; idle_timeout=0.0)
        close(srv)
        srv,url = serve_chunks() do http
            write(http, chatchunk("a")); flush(http)
            write(http, chatchunk("b")); flush(http)
            write(http, donechunk()); flush(http)
        end
        cb_plain = HttpStreamCallback(out=IOBuffer(), schema=ChatCompletionSchema())
        streamed_request!(cb_plain, url, ["Content-Type"=>"application/json"], "{}")
        close(srv)
        @test length(cb0.chunks) == length(cb_plain.chunks)
        @test length(cb0.chunks) >= 2
    end

    @testset "isolated abort-detection (helper alone, no HTTP teardown)" begin
        # Time ONLY the helper read+abort, not HTTP cleanup. Handler parks on a
        # Condition (no sleep) so it can't inflate the measured latency.
        done = Ref(false)
        srv,url = serve_chunks() do http
            write(http, chatchunk("x")); flush(http)
            try
                while !done[] && isopen(http); sleep(0.05); end
            catch; end
        end
        try
            HTTP.open("POST", url, ["Content-Type"=>"application/json"];
                      retry=false, reuse_limit=0) do stream
                write(stream, "{}"); HTTP.closewrite(stream)
                HTTP.startread(stream)
                readavailable(stream)  # consume the first chunk
                fired = Ref(false)
                t0 = time()
                nbytes = 0; threw = false
                try
                    data = readavailable_with_idle_timeout(stream, 0.3; fired)
                    nbytes = length(data)
                catch
                    threw = true   # abort can surface as EOFError from the closed socket
                end
                dt = time() - t0
                @test fired[] == true   # the timer fired regardless of empty-return vs throw
                @printf("    helper abort fired in %.3fs (timeout=0.3s, %s, %d bytes)\n",
                        dt, threw ? "threw EOFError" : "returned empty", nbytes)
                @test dt < 1.5
                # re-throw so HTTP.open doesn't try to keep draining a dead conn
                throw(StreamIdleTimeoutError(0.3))
            end
        catch e
            # HTTP.open re-wraps the deliberate socket kill as EOFError/RequestError
            (e isa StreamIdleTimeoutError || e isa EOFError ||
             e isa HTTP.RequestError || e isa Base.IOError) || rethrow()
        end
        done[] = true
        close(srv)
    end
end
