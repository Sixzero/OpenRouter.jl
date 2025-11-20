using OpenRouter
using Documenter

DocMeta.setdocmeta!(OpenRouter, :DocTestSetup, :(using OpenRouter); recursive=true)

makedocs(;
    modules=[OpenRouter],
    authors="SixZero <havliktomi@hotmail.com> and contributors",
    sitename="OpenRouter.jl",
    format=Documenter.HTML(;
        canonical="https://sixzero.github.io/OpenRouter.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Sixzero/OpenRouter.jl",
    devbranch="master",
)
