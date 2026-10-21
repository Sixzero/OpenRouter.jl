using Test, JSON3
using OpenRouter: HttpStreamHooks, StreamChunk, record_chunk_provenance!

@testset "stream provenance" begin
    chunk(s) = StreamChunk(data=s, json=JSON3.read(s))
    p = Dict{String,Any}()
    record_chunk_provenance!(p, chunk("""{"id":"gen-1","model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[]}"""))
    record_chunk_provenance!(p, chunk("""{"id":"gen-2","provider":"Other","choices":[]}"""))
    @test p == Dict("responseId"=>"gen-1", "servedModel"=>"anthropic/claude-haiku-4.5", "upstream"=>"Amazon Bedrock")
    a = Dict{String,Any}()
    record_chunk_provenance!(a, chunk("""{"type":"message_start","message":{"id":"msg_bdrk_x","model":"anthropic.claude-haiku-4-5"}}"""))
    @test a == Dict("responseId"=>"msg_bdrk_x", "servedModel"=>"anthropic.claude-haiku-4-5")
    g = Dict{String,Any}()
    record_chunk_provenance!(g, chunk("""{"responseId":"r1","modelVersion":"gemini-x","candidates":[]}"""))
    @test g == Dict("responseId"=>"r1", "servedModel"=>"gemini-x")
    o = Dict{String,Any}()
    record_chunk_provenance!(o, chunk("""{"model":"llama3","message":{"role":"assistant","content":"hi"},"done":false}"""))
    @test o == Dict("servedModel"=>"llama3")
    l = Dict{String,Any}()
    record_chunk_provenance!(l, chunk("""{"id":"c1","model":null,"choices":[]}"""))
    record_chunk_provenance!(l, chunk("""{"id":"c1","model":"gpt-x","choices":[]}"""))
    @test l == Dict("responseId"=>"c1", "servedModel"=>"gpt-x")
end
