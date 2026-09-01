using Test
using OpenRouter
using Sockets

@testset "stream_cancel_flag fast cancel" begin
    sse_chunk(s) = string(string(sizeof(s), base=16), "\r\n", s, "\r\n")
    const_headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
    body_chunk = sse_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n")
    done_chunk = sse_chunk("data: [DONE]\n\n") * "0\r\n\r\n"

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

    function request_stream(port; cancel_flag)
        cb = OpenRouter.HttpStreamHooks(schema=OpenRouter.ChatCompletionSchema(), out=devnull)
        OpenRouter.streamed_request!(cb, "http://127.0.0.1:$port/v1/x", ["Content-Type" => "application/json"], "{}";
            stream_idle_timeout=300.0, stream_first_chunk_timeout=60.0, stream_cancel_flag=cancel_flag, retry=false)
    end

    # Warm-up: compile the request path so timing asserts measure policy, not JIT.
    let
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk, done_chunk)
            sleep(0.5); close(sock)
        end
        try request_stream(port; cancel_flag=Threads.Atomic{Bool}(false)) catch end
        teardown()
    end

    @testset "cancel during byte-silent stream → InterruptException, fast" begin
        # Upstream sends headers then goes silent — the per-chunk cancel checks
        # in caller formatters would never run; only the watcher can unblock.
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers)
            sleep(30)
        end
        flag = Threads.Atomic{Bool}(false)
        @async (sleep(0.5); flag[] = true)
        t = @elapsed err = try request_stream(port; cancel_flag=flag); nothing catch e; e end
        teardown()
        @test err isa InterruptException
        @test t < 5  # NOT the 60s first-chunk or 300s idle timeout
    end

    @testset "cancel mid-stream between chunks → InterruptException, fast" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk)  # healthy start
            sleep(30)                               # stalls mid-generation
        end
        flag = Threads.Atomic{Bool}(false)
        @async (sleep(0.5); flag[] = true)
        t = @elapsed err = try request_stream(port; cancel_flag=flag); nothing catch e; e end
        teardown()
        @test err isa InterruptException
        @test t < 5
    end

    @testset "flag already set at request start → cancels on first watcher tick" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers)
            sleep(30)
        end
        flag = Threads.Atomic{Bool}(true)  # pre-set: Stop pressed before dispatch
        t = @elapsed err = try request_stream(port; cancel_flag=flag); nothing catch e; e end
        teardown()
        @test err isa InterruptException
        @test t < 5
    end

    @testset "unset flag never interferes with a healthy stream" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk)
            sleep(0.5)
            write(sock, done_chunk)
            sleep(0.5)
            close(sock)
        end
        flag = Threads.Atomic{Bool}(false)
        err = try request_stream(port; cancel_flag=flag); nothing catch e; e end
        teardown()
        @test err === nothing
    end

    @testset "no flag (nothing) — path unchanged" begin
        port, teardown = serve() do sock
            readavailable(sock)
            write(sock, const_headers, body_chunk, done_chunk)
            sleep(0.5)
            close(sock)
        end
        err = try request_stream(port; cancel_flag=nothing); nothing catch e; e end
        teardown()
        @test err === nothing
    end
end
