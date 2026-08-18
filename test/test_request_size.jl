using Test
using JSON3
using OpenRouter: request_size_report, _snippet, _fmt_bytes

@testset "request size reporting" begin
    @testset "byte formatting" begin
        @test _fmt_bytes(512) == "512B"
        @test _fmt_bytes(2048) == "2.0KB"
        @test _fmt_bytes(3 * 1024^2) == "3.0MB"
    end

    @testset "breakdown, biggest first" begin
        body = JSON3.write(Dict(
            "model" => "x",
            "system" => repeat("s", 3000),
            "messages" => [
                Dict("role" => "user", "content" => repeat("u", 9000)),
                Dict("role" => "assistant", "content" => "hi"),
            ],
        ))
        rep = request_size_report(body)
        @test occursin("total=", rep)
        # user message is the biggest part, so it must lead the breakdown
        @test occursin(r"top: user\[1\]=8\.\dKB", rep)
        @test occursin("system=2.9KB", rep)
    end

    @testset "gemini contents / responses input" begin
        gem = JSON3.write(Dict("contents" => [Dict("role" => "user", "parts" => [Dict("text" => repeat("g", 2000))])]))
        @test occursin("user[1]=", request_size_report(gem))
        resp = JSON3.write(Dict("input" => [Dict("role" => "user", "content" => repeat("r", 2000))]))
        @test occursin("user[1]=", request_size_report(resp))
    end

    @testset "non-JSON body degrades to total only" begin
        rep = request_size_report("not json at all")
        @test rep == "total=15B"
    end

    @testset "snippet is unicode-safe" begin
        body = repeat("á", 5000)  # multibyte: naive body[1:500] would throw
        s = _snippet(body)
        @test length(s) == 500
        @test _snippet("short") == "short"
    end
end
