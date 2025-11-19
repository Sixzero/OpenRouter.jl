using Test
using OpenRouter
using OpenRouter.Streaming

@testset "Streaming Interface" begin
    @testset "StreamChunk" begin
        chunk = StreamChunk(event=:test, data="test data", json=nothing)
        @test chunk.event == :test
        @test chunk.data == "test data"
        @test chunk.json === nothing
        
        # Test show method
        io = IOBuffer()
        show(io, chunk)
        output = String(take!(io))
        @test occursin("StreamChunk", output)
        @test occursin("test data", output)
    end
    
    @testset "HttpStreamCallback" begin
        callback = HttpStreamCallback()
        @test callback.out == stdout
        @test callback.schema === nothing
        @test isempty(callback.chunks)
        @test !callback.verbose
        
        # Test basic operations
        chunk = StreamChunk(data="test")
        push!(callback, chunk)
        @test length(callback) == 1
        @test !isempty(callback)
        
        empty!(callback)
        @test isempty(callback)
    end
    
    @testset "Schema-based streaming methods" begin
        schema = ChatCompletionSchema()
        
        # Test extract_chunks
        blob = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
        chunks, spillover = Streaming.extract_chunks(schema, blob)
        @test length(chunks) == 1
        @test chunks[1].data == "{\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}"
        @test spillover == ""
        
        # Test is_done
        done_chunk = StreamChunk(data="[DONE]")
        @test Streaming.is_done(schema, done_chunk)
        
        not_done_chunk = StreamChunk(data="{\"choices\":[]}")
        @test !Streaming.is_done(schema, not_done_chunk)
        
        # Test extract_content
        json_data = JSON3.read("{\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}")
        content_chunk = StreamChunk(json=json_data)
        content = Streaming.extract_content(schema, content_chunk)
        @test content == "Hello"
    end
    
    @testset "Print content methods" begin
        # Test IO output
        io = IOBuffer()
        Streaming.print_content(io, "test content")
        @test String(take!(io)) == "test content"
        
        # Test Channel output
        ch = Channel{String}(10)
        Streaming.print_content(ch, "test content")
        @test take!(ch) == "test content"
        close(ch)
        
        # Test nothing output
        @test Streaming.print_content(nothing, "test content") === nothing
    end

    @testset "Gemini SSE CRLF handling" begin
        schema = GeminiSchema()
        blob = "data: {\"candidates\": [{\"finishReason\": \"STOP\"}]}\r\n\r\n"
        chunks, spill = Streaming.extract_chunks(schema, blob)
        @test length(chunks) == 1
        @test spill == ""
        @test chunks[1].json[:candidates][1][:finishReason] == "STOP"
        @test Streaming.is_done(schema, chunks[1])
    end
end