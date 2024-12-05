import graphviz

def main() -> None:
        graph: graphviz.Digraph = graphviz.Digraph("graph", node_attr = {"shape": "box"})
        functions: list[str] = [
                "make.cvgroup",
                "make.cvgroup.balanced",
                "cross.fit.propensities",
                "optimCutHajek",
                "constregn",
                "boostregn",
                "svmregn",
                "forestregn",
                "linregn",
                "constquant",
                "linquant",
                "forestquant",
                "forestconddist",
                "kernelconddist",
                "getprop",
                "binaryExtrapolationquant",
                "binaryExtrapolationKappa",
                "bootstrapProps",
                "summarizeBoots",
                "summarizeResults",
                "dvds",
                "zsb",
                "qb"
        ]
        nodes_labels: dict[str, str] = {chr(ord("@") + node + 1): label for node, label in enumerate(functions)}
        for node, label in nodes_labels.items():
                graph.node(node, label = label)
        graph.edges([
                "AB",
                "JK",
                "KL",
                "PQ",
                "BR",
                "CR",
                "BU",
                "CU",
                "OR",
                "OU",
                "PU",
                "QU",
                "RU",
                "SU",
                "TU",
                "BV",
                "CV",
                "DV",
                "OV",
                "RV",
                "SV",
                "TV",
                "BW",
                "CW",
                "OW",
                "PW",
                "RW",
                "SW",
                "TW"
        ])
        graph.render(directory = "assets", format = "png")
        return

if __name__ == "__main__":
        main()

