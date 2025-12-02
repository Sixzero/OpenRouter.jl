# Precompilation workload using echo server

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        with_echo_server() do
            for p in ECHO_PROVIDERS
                aigen(; prompt="hi", model="$p:test")
            end
        end
    end
end
