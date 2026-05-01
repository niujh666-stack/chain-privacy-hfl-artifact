from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="CPA-LPPF-HFL command line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_attack = sub.add_parser("attack", help="run attack comparison")
    p_attack.add_argument("--config", required=True)
    p_attack.add_argument("--output", required=True)

    p_matrix = sub.add_parser("defense-matrix", help="run defense by attack matrix")
    p_matrix.add_argument("--config", required=True)
    p_matrix.add_argument("--output", required=True)

    p_plot = sub.add_parser("plot", help="plot results")
    p_plot.add_argument("--input", required=True)
    p_plot.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "attack":
        from cpa_lppf.experiments.run_attack import main as run_attack

        run_attack(["--config", args.config, "--output", args.output])
    elif args.command == "defense-matrix":
        from cpa_lppf.experiments.run_defense_matrix import main as run_matrix

        run_matrix(["--config", args.config, "--output", args.output])
    elif args.command == "plot":
        from cpa_lppf.analysis.plot_paper_figures import main as run_plot

        run_plot(["--input", args.input, "--output", args.output])


if __name__ == "__main__":
    main()
