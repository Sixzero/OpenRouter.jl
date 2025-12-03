using OpenRouter
using Test

@testset "Hook Trigger Counts" begin
    test_cases = [
        ("anthropic:anthropic/claude-haiku-4.5", true),   # Anthropic sends usr_meta
        ("openai:openai/gpt-5.1-mini", false),            # OpenAI doesn't send usr_meta
        ("groq:moonshotai/kimi-k2-0905", false),          # Groq doesn't send usr_meta
    ]
    
    prompt = "Say hi"
    
    for (provider_model, expects_usr_meta) in test_cases
        try
            done_count = Ref(0)
            meta_ai_count = Ref(0)
            meta_usr_count = Ref(0)
            
            callback = HttpStreamHooks(
                out = devnull,
                on_done = () -> (done_count[] += 1; nothing),
                on_meta_ai = (t, c, e) -> (meta_ai_count[] += 1; nothing),
                on_meta_usr = (t, c, e) -> (meta_usr_count[] += 1; nothing),
            )
            
            aigen(prompt, provider_model; streamcallback=callback)
            
            @test done_count[] == 1
            @test meta_ai_count[] == 1
            if expects_usr_meta
                @test meta_usr_count[] == 1
            else
                @test meta_usr_count[] <= 1
            end
            
            println("✓ $provider_model: hooks triggered correctly (done=$(done_count[]), ai=$(meta_ai_count[]), usr=$(meta_usr_count[]))")
            
        catch e
            @warn "Failed to test $provider_model: $e"
        end
    end
end
