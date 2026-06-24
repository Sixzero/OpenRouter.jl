#!/usr/bin/env julia
# Isolate the abort LATENCY: how fast does closing the underlying socket
# unblock a readavailable that's parked waiting for bytes?
# Layer 1: raw TCPSocket.  Layer 2: HTTP.Stream (full streamed_request! path).
using HTTP, Sockets, Printf

println("="^70)
println("LAYER 1: raw TCPSocket — close from a Timer while readavailable blocks")
println("="^70)
let
    server = listen(Sockets.localhost, 0)
    port = getsockname(server)[2]
    @async begin
        sock = accept(server)
        write(sock, "hello"); flush(sock)
        sleep(30)  # then stall, keep socket open
    end
    cli = connect(Sockets.localhost, port)
    readavailable(cli)  # consume "hello"
    fired = Ref(false)
    timer = Timer(0.5) do _
        fired[] = true
        close(cli)
    end
    t0 = time()
    local data
    try
        data = readavailable(cli)  # blocks; timer closes cli to unblock
    catch e
        data = UInt8[]
        @printf("  readavailable threw %s\n", typeof(e))
    end
    dt = time() - t0
    close(timer)
    @printf("  raw socket: unblocked in %.3fs after 0.5s timer fired=%s (%d bytes)\n",
            dt, fired[], length(data))
    close(server)
end

println()
println("="^70)
println("LAYER 2: HTTP.Stream — close stream.stream (Connection) from Timer")
println("="^70)
let
    server = HTTP.serve!("127.0.0.1", 0; stream=true, listenany=true) do http
        while !eof(http); readavailable(http); end
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        write(http, "data: x\n\n"); flush(http)
        sleep(30)
    end
    port = HTTP.Sockets.getsockname(server.listener.server)[2]
    url = "http://127.0.0.1:$port"

    HTTP.open("POST", url, ["Content-Type"=>"application/json"]; retry=false, reuse_limit=0) do stream
        write(stream, "{}"); HTTP.closewrite(stream)
        HTTP.startread(stream)
        readavailable(stream)  # consume "data: x"
        fired = Ref(false)
        underlying = hasproperty(stream, :stream) ? stream.stream : stream
        @printf("  underlying type: %s\n", typeof(underlying))
        timer = Timer(0.5) do _
            fired[] = true
            close(underlying)
        end
        t0 = time()
        local data
        try
            data = readavailable(stream)
        catch e
            data = UInt8[]
            @printf("  readavailable threw %s\n", typeof(e))
        end
        dt = time() - t0
        close(timer)
        @printf("  HTTP.Stream: unblocked in %.3fs after 0.5s timer fired=%s (%d bytes)\n",
                dt, fired[], length(data))
    end
    close(server)
end
