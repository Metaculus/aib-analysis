from aib_analysis.math.stats import ConfidenceIntervalCalculator, MeanHypothesisCalculator, ObservationStats

import pytest

# NOTE: tests were copied from forecasting-tools stats testing

class TestConfidenceInterval:

    def test_confidence_interval_from_mean_and_std_v1(self) -> None:
        # https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample/estimating-population-mean/v/calculating-a-one-sample-t-interval-for-a-mean
        num_observations = 14
        mean = 700
        std = 50
        confidence = 0.95

        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_mean_and_std(
                mean, std, num_observations, confidence
            )
        )

        assert confidence_interval.mean == mean
        assert confidence_interval.margin_of_error == pytest.approx(28.9, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(671.1, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(728.9, 0.1)

    def test_confidence_interval_from_mean_andstd_v2(self) -> None:
        # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Introductory_Statistics_1e_(OpenStax)/08%3A_Confidence_Intervals/8.E%3A_Confidence_Intervals_(Exercises)
        num_observations = 100
        mean = 23.6
        std = 7
        confidence = 0.95

        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_mean_and_std(
                mean, std, num_observations, confidence
            )
        )

        assert confidence_interval.margin_of_error == pytest.approx(1.372, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(22.228, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(24.972, 0.1)

    def test_confidence_interval_from_observations_v1(self) -> None:
        # https://www.statskingdom.com/confidence-interval-calculator.html
        data: list[float] = [
            2.016958256852196,
            -2.1545547595542955,
            0.17643058731468048,
            -0.17899745527693173,
            -1.4400636862377763,
            1.5761768611550813,
            -0.1980518250021597,
            -0.011601732681319138,
            -1.7437464244027827,
            -0.3944474061704416,
            0.43389005591630586,
            1.1077540943600064,
            -0.6687719492567181,
            -0.9757879464441855,
            -0.5618087528418959,
            -0.9865765689103303,
            -0.7048146469454047,
            2.612506442801254,
            0.808876533367124,
            -0.13846324336650978,
            -0.5766259715841434,
            -0.29767038112826727,
            0.06583008361659538,
            0.7460570036633725,
            -0.640818986028395,
        ]
        confidence = 0.9
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(-0.08513, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(0.3818, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(-0.4669, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(0.2967, 0.1)

    @pytest.mark.skip(
        reason="all 3 of my code, the online stat calculator, and gpt analysis calculate std as 829k which different from than the textbook 909k. The textbook also seems to be using z scores (t is better here?)"
    )
    def test_confidence_interval_from_observations_v2(self) -> None:
        # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Introductory_Statistics_1e_(OpenStax)/08%3A_Confidence_Intervals/8.E%3A_Confidence_Intervals_(Exercises)
        data: list[float] = [
            3600,
            1243900,
            10900,
            385200,
            581500,
            7400,
            2900,
            400,
            3714500,
            632500,
            391000,
            467400,
            56800,
            5800,
            405200,
            733200,
            8000,
            468700,
            75200,
            41000,
            13300,
            9500,
            953800,
            1113500,
            1109300,
            353900,
            986100,
            88600,
            378200,
            13200,
            3800,
            745100,
            5800,
            3072100,
            1626700,
            512900,
            2309200,
            6600,
            202400,
            15800,
        ]
        confidence = 0.95
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(568_873, 0.1)
        assert confidence_interval.standard_deviation == pytest.approx(909_200)
        assert confidence_interval.margin_of_error == pytest.approx(281_764)
        assert confidence_interval.lower_bound == pytest.approx(287_109)
        assert confidence_interval.upper_bound == pytest.approx(850_637)

    def test_confidence_interval_from_observations_v3(self) -> None:
        # https://stats.libretexts.org/Workbench/PSYC_2200%3A_Elementary_Statistics_for_Behavioral_and_Social_Science_(Oja)_WITHOUT_UNITS/08%3A_One_Sample_t-test/8.05%3A_Confidence_Intervals/8.5.01%3A_Practice_with_Confidence_Interval_Calculations
        data: list[float] = [
            8.6,
            9.4,
            7.9,
            6.8,
            8.3,
            7.3,
            9.2,
            9.6,
            8.7,
            11.4,
            10.3,
            5.4,
            8.1,
            5.5,
            6.9,
        ]
        confidence = 0.95
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(8.2267, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(0.924, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(7.3, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(9.15, 0.1)

    def test_confidence_interval_from_observations_v4(self) -> None:
        # https://stats.libretexts.org/Workbench/PSYC_2200%3A_Elementary_Statistics_for_Behavioral_and_Social_Science_(Oja)_WITHOUT_UNITS/08%3A_One_Sample_t-test/8.05%3A_Confidence_Intervals/8.5.01%3A_Practice_with_Confidence_Interval_Calculations
        data: list[float] = [
            79,
            145,
            147,
            160,
            116,
            100,
            159,
            151,
            156,
            126,
            137,
            83,
            156,
            94,
            121,
            144,
            123,
            114,
            139,
            99,
        ]
        confidence = 0.9
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(127.45, 0.1)
        assert confidence_interval.standard_deviation == pytest.approx(25.965, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(10.038, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(117.412, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(137.488, 0.1)

    def test_non_normal_data_errors(self) -> None:
        with pytest.raises(ValueError):
            data: list[float] = [1, 5, 8, 7, 79, 3, 45, 67, 43, 65, 87, 12]
            confidence = 0.9
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )

    def test_insufficient_data_errors(self) -> None:
        with pytest.raises(Exception):
            data: list[float] = []
            confidence = 0.9
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )


class TestMeanStatCalculator:

    def test_mean_is_greater_than_hypothesis_mean(self) -> None:
        # https://ecampusontario.pressbooks.pub/introstats/chapter/8-7-hypothesis-tests-for-a-population-mean-with-unknown-population-standard-deviation/
        observations = [65.0, 67.0, 66.0, 68.0, 72.0, 65.0, 70.0, 63.0, 63.0, 71.0]
        hypothesis_mean = 65.0
        confidence = 0.99

        hypothesis_test = (
            MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
                observations, hypothesis_mean, confidence
            )
        )

        assert hypothesis_test.p_value == pytest.approx(0.0396, 0.01)
        assert hypothesis_test.hypothesis_rejected == False


    def test_mean_is_equal_to_hypothesis_mean(self) -> None:
        # https://ecampusontario.pressbooks.pub/introstats/chapter/8-7-hypothesis-tests-for-a-population-mean-with-unknown-population-standard-deviation/
        hypothesis_mean = 3.78
        count = 100
        average = 3.62
        standard_deviation = 0.7
        confidence = 0.95

        observation_stats = ObservationStats(
            average=average,
            standard_deviation=standard_deviation,
            count=count,
        )

        hypothesis_test = MeanHypothesisCalculator._test_if_mean_is_equal_to_than_hypothesis_mean_w_observation_stats(
            observation_stats, hypothesis_mean, confidence
        )

        assert hypothesis_test.p_value == pytest.approx(0.0244, 0.01)
        assert hypothesis_test.hypothesis_rejected == True