"""
Changes the forecast csv so there are only rows for the control bots.
"""
import pandas as pd

control_bots = [
    "mf-bot-1",
    "mf-bot-3",
]
def main() -> None:
    input_file = "local/q3_archive/bots_score_data_q3.csv"
    pro_file = "local/q3_archive/pros_score_data_q3.csv"
    output_file = "local/bots_score_data_q3_control.csv"

    pro_question_titles: list[str] = list(pd.read_csv(pro_file)["question_title"].unique())

    df = pd.read_csv(input_file)
    df = df[df["forecaster"].isin(control_bots)]
    # df = df[df["question_title"].isin(pro_question_titles)]
    df.to_csv(output_file, index=False)

    pass

if __name__ == "__main__":
    main()