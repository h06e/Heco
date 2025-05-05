using Heco
using Documenter

DocMeta.setdocmeta!(Heco, :DocTestSetup, :(using Heco); recursive=true)

makedocs(;
    modules=[Heco],
    authors="Robin Valmalette",
    sitename="Heco.jl",
    format=Documenter.HTML(;
        canonical="https://h06e.github.io/Heco.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/h06e/Heco.jl",
    devbranch="main",
)
