#!/usr/bin/env julia
# Allocation-focused head-to-head of the SSE idle-timeout designs.
# Wall-time per chunk is network-bound (~50µs loopback round-trip), so the
# mechanism's overhead is sub-1% and lost in noise — we report it but the
# DETERMINISTIC signal is bytes-allocated per chunk.
using HTTP, Sockets, Printf

_abort_read!(stream::HTTP.Stream) =
    close(hasproperty(stream, :stream) ? stream.stream : stream)
_abort_read!(stream) = close(stream)

# ---- design #3 (current) ----
function read_d3(stream, timeout::Real; fired::Ref{Bool}=Ref(false))
    timeout <= 0 && return readavailable(stream)
    timer = Timer(timeout) do _
        fired[] = true
        try _abort_read!(stream) catch end
    end
    try
        return readavailable(stream)
    finally
        close(timer)
    end
end

# ---- design #1 (@async per read) ----
function read_d1(stream, timeout::Real; fired::Ref{Bool}=Ref(false))
    timeout <= 0 && return readavailable(stream)
    reader = @async readavailable(stream)
    timer = Timer(timeout) do _
        fired[] = true
        try _abort_read!(stream) catch end
    end
    try
        return fetch(reader)
    finally
        close(timer)
    end
end

# ---- design #2 (StreamWatchdog: one polling timer per stream) ----
mutable struct StreamWatchdog
    stream::Any; timeout::Float64; last::Float64; fired::Bool; timer::Union{Timer,Nothing}
end
function arm!(wd::StreamWatchdog)
    interval = clamp(wd.timeout/4, 0.05, 5.0)
    wd.last = time()
    wd.timer = Timer(interval; interval=interval) do _
        if time() - wd.last >= wd.timeout
            wd.fired = true; try _abort_read!(wd.stream) catch end
        end
    end
    wd
end
bump!(wd::StreamWatchdog) = (wd.last = time(); nothing)
disarm!(wd::StreamWatchdog) = (wd.timer !== nothing && close(wd.timer); nothing)

function make_server(nchunks::Int; payload::String="data: x\n\n")
    server = HTTP.serve!("127.0.0.1", 0; stream=true, listenany=true) do http
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for _ in 1:nchunks
            write(http, payload); flush(http)
        end
    end
    port = HTTP.Sockets.getsockname(server.listener.server)[2]
    (server, "http://127.0.0.1:$port")
end

# Drain with a per-read strategy fn (gets the stream, returns bytes).
function drain(url, mk_reader)
    nb = 0; nc = 0
    HTTP.open("GET", url) do stream
        HTTP.startread(stream)
        rf = mk_reader(stream)
        while !eof(stream)
            data = rf()
            nb += length(data); nc += 1
        end
    end
    (nb, nc)
end

# mk_reader: stream -> (() -> bytes). Lets design#2 arm a per-stream watchdog.
strat_baseline() = stream -> () -> readavailable(stream)
strat_d1()       = stream -> () -> read_d1(stream, 300.0)
strat_d3()       = stream -> () -> read_d3(stream, 300.0)
function strat_d2()
    stream -> begin
        wd = arm!(StreamWatchdog(stream, 300.0, time(), false, nothing))
        () -> (bump!(wd); d = readavailable(stream); eof(stream) && disarm!(wd); d)
    end
end

function measure(label, N, strat; reps=8)
    let (srv,url)=make_server(N); drain(url, strat()); close(srv); end  # warmup
    times=Float64[]; allocs=Float64[]; bytes=0; chunks=0
    for _ in 1:reps
        (srv,url)=make_server(N)
        a = @allocated ((bytes,chunks)=drain(url, strat()))
        close(srv)
        (srv,url)=make_server(N)
        t = @elapsed drain(url, strat())
        close(srv)
        push!(allocs,a); push!(times,t)
    end
    sort!(times); sort!(allocs)
    tmed=times[cld(end,2)]; amed=allocs[cld(end,2)]
    @printf("  %-16s N=%-5d  %7.1f ns/chunk  %8.0f B  %7.1f B/chunk\n",
            label, N, tmed/chunks*1e9, amed, amed/chunks)
    (;label,N,t=tmed,tper=tmed/chunks,a=amed,aper=amed/chunks,chunks)
end

println("="^92)
println("ALLOC-FOCUSED PER-CHUNK BENCH (fast local SSE, small chunks, no sleeps; median of 8 reps)")
println("="^92)
R=[]
for N in (50,500,2000)
    println("--- N=$N ---")
    push!(R, measure("baseline",        N, strat_baseline))
    push!(R, measure("design#1 @async", N, strat_d1))
    push!(R, measure("design#2 wd",     N, strat_d2))
    push!(R, measure("design#3 cur",    N, strat_d3))
end
println("="^92)
println("Δ vs baseline (B/chunk is the deterministic signal; ns/chunk is network-noisy)")
println("="^92)
for N in (50,500,2000)
    b=first(r for r in R if r.label=="baseline" && r.N==N)
    for lbl in ("design#1 @async","design#2 wd","design#3 cur")
        d=first(r for r in R if r.label==lbl && r.N==N)
        @printf("  N=%-5d %-16s  Δ %+8.1f B/chunk   Δ %+8.1f ns/chunk (%.2f%% wall)\n",
                N,lbl,d.aper-b.aper,(d.tper-b.tper)*1e9,(d.t-b.t)/b.t*100)
    end
end
