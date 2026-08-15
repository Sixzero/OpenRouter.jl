using Test
using OpenRouter
using Sockets

@testset "First-chunk / idle stream timeouts" begin
    # SSE chunked-encoding helper
    sse_chunk(s) = string(string(sizeof(s), base=16), "\r\n", s, "\r\n")
    const_headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
    body_chunk = sse_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n")
    done_chunk = sse_chunk("data: [DONE]\n\n") * "0\r\n\r\n"

    # Serve one connection with `handler(sock)`; track sockets so teardown closes them.
    function serve(handler)
        server = listen(IPv4(127, 0, 0, 1), 0)
        port = Int(getsockname(server)[2])
        socks = []
        @async while isopen(server)
            sock = try accept(server) catch; break end
            push!(socks, sock)
            @async try handler(sock) catch end
        end
        teardown() = (close(server); foreach(s -> try close(s) catch end, socks))
        return port, teardown
    end

    function request_stream(port; first_chunk_timeout, idle_timeout)
        cb = OpenRouter.HttpStreamHooks(schema=OpenRouter.ChatCompletionSchema(), throw_on_error=true, out=devnull)
        OpenRouter.streamed_request!(cb, "http://127.0.0.1:$port/v1/x", ["Content-Type" => "application/json"], "{}";
            stream_idle_timeout=idle_timeout, stream_first_chunk_timeout=first_chunk_timeout, retry=false)
    end

    function timed_error(port; kw...)
        t = @elapsed err = try request_stream(port; kw...); nothing catch e; e end
        (t, err)
    end

    # Warm-up: compile the whole request path once so timing asserts below
    # measure the timeout policy, not first-call JIT latency.
    let
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk, done_chunk)
            sleep(0.5); close(sock)
        end
        try request_stream(port; first_chunk_timeout=5.0, idle_timeout=300.0) catch end
        teardown()
    end

    @testset "headers sent, body never arrives → StreamIdleTimeoutError" begin
        port, teardown = serve() do sock
            readavailable(sock); write(sock, const_headers); sleep(30)
        end
        t, err = timed_error(port; first_chunk_timeout=1.0, idle_timeout=300.0)
        teardown()
        @test err isa OpenRouter.StreamIdleTimeoutError
        @test err.timeout == 1.0
        @test t < 5  # first-chunk deadline, not the 300s idle timeout
    end

    @testset "connection accepted, headers never sent → StreamIdleTimeoutError" begin
        port, teardown = serve() do sock
            readavailable(sock); sleep(30)
        end
        t, err = timed_error(port; first_chunk_timeout=1.0, idle_timeout=300.0)
        teardown()
        @test err isa OpenRouter.StreamIdleTimeoutError
        @test err.timeout == 1.0
        @test t < 5
    end

    @testset "first-chunk deadline is ABSOLUTE across headers+body (no per-call reset)" begin
        # Headers eat most of the deadline; body silence must be caught by the
        # REMAINING time, not a fresh full window.
        port, teardown = serve() do sock
            readavailable(sock)
            sleep(1.0)                     # ~half the 2s deadline spent on headers
            write(sock, const_headers)
            sleep(30)                      # body never arrives
        end
        t, err = timed_error(port; first_chunk_timeout=2.0, idle_timeout=300.0)
        teardown()
        @test err isa OpenRouter.StreamIdleTimeoutError
        @test t < 3.5  # ≈2s total, NOT 1s headers + fresh 2s body window
    end

    @testset "slow-but-healthy stream survives first-chunk deadline" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers)
            sleep(0.3)              # first chunk within deadline
            write(sock, body_chunk)
            sleep(2.0)              # gap LONGER than first-chunk timeout — idle policy active now
            write(sock, done_chunk)
            sleep(0.5)              # let client consume before close (avoid close/read race)
            close(sock)
        end
        err = timed_error(port; first_chunk_timeout=1.0, idle_timeout=300.0)[2]
        teardown()
        @test err === nothing
    end

    @testset "first_chunk_timeout=0 disables first phase (slow first chunk OK)" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers)
            sleep(1.5)              # would trip a 1s first-chunk deadline if wrongly active
            write(sock, body_chunk, done_chunk)
            sleep(0.5)              # let client consume before close (avoid close/read race)
            close(sock)
        end
        err = timed_error(port; first_chunk_timeout=0.0, idle_timeout=300.0)[2]
        teardown()
        @test err === nothing
    end

    @testset "idle timeout fires after first chunk" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk)  # healthy start
            sleep(30)                               # then stalls mid-stream
        end
        t, err = timed_error(port; first_chunk_timeout=5.0, idle_timeout=1.0)
        teardown()
        @test err isa OpenRouter.StreamIdleTimeoutError
        @test err.timeout == 1.0
        @test t < 5
    end

    @testset "error status with silent body → StreamIdleTimeoutError, not hang" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, "HTTP/1.1 500 Internal Server Error\r\nTransfer-Encoding: chunked\r\n\r\n")
            sleep(30)  # error body never arrives
        end
        t, err = timed_error(port; first_chunk_timeout=1.0, idle_timeout=300.0)
        teardown()
        @test err isa OpenRouter.StreamIdleTimeoutError
        @test t < 5
    end

    @testset "StreamIdleTimeoutError message is transient-retryable" begin
        # EasyContext._is_transient_error matches on "timeout" in showerror output —
        # keep that contract (retry loop must classify a stalled stream as transient).
        msg = lowercase(sprint(showerror, OpenRouter.StreamIdleTimeoutError(60.0)))
        @test occursin("timeout", msg)
    end
end
