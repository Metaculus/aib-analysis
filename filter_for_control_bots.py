"""
Changes the forecast csv so there are only rows for the control bots.
"""
import pandas as pd

control_bots = [
    "mf-bot-1",
    "mf-bot-3",
]
def main() -> None:
    input_file = "local/archived_scores/bots_score_data_q4.csv"
    output_file = "local/bots_score_data_q4_control.csv"

    df = pd.read_csv(input_file)
    df = df[df["forecaster"].isin(control_bots)]
    df.to_csv(output_file, index=False)

    pass

if __name__ == "__main__":
    main()