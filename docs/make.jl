using SmoreFit
using Documenter

DocMeta.setdocmeta!(SmoreFit, :DocTestSetup, :(using SmoreFit); recursive=true)

makedocs(;
    modules=[SmoreFit],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SmoreFit.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/SmoreFit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/SmoreFit.jl",
    devbranch="main",
)
