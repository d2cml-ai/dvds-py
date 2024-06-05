import graphviz

def main() -> None:
        graph: graphviz.Digraph = graphviz.Digraph("simplified")
        functions: list[str] = [
                "phi.continuous",
                "dvds.binary",
                "dvds.continuous",
                "extrema",
                "extrema.os",
                "extrema.aipw"
        ]
        nodes_labels: dict[str, str] = {chr(ord("@") + node + 1): label for node, label in enumerate(functions)}
        for node, label in nodes_labels.items():
                graph.node(node, label = label)
        graph.edges([
                "AC",
                "DE",
                "EF"
        ])
        graph.render(directory = "assets", format = "png")
        return

if __name__ == "__main__":
        main()