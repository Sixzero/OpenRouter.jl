# Precompilation workload using echo server

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        try
            with_echo_server() do
                for p in ECHO_PROVIDERS
                    aigen(; prompt="hi", model="$p:test")
                end
            end
        catch e
            e isa Base.IOError && contains(e.msg, "address already in use") && @debug "Skipping precompilation: echo server port in use"
        end
    end
end
