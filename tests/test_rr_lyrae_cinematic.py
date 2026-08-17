"""Tests for the RR Lyrae cinema (#160).

The film claims four things: that three real light curves fold on their
catalogue periods, that Bailey type is period and amplitude, that Wesenheit
tightens the RRab ridge, and that the Magellanic gap is a distance still
missing a metallicity term. Each one is re-derived from the committed CSVs.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.rr_lyrae_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_CATALOG_CSV,
    DEFAULT_LIGHTCURVE_CSV,
    HERO_STARS,
    LMC_DISTANCE_MODULUS,
    MINIMUM_POINTS_TO_FIT,
    PUBLISHED_SMC_DISTANCE_MODULUS,
    WESENHEIT_COLOR_COEFFICIENT,
    RrLyraeCinematicAnimator,
    buildHeroCurves,
    distanceKiloparsecs,
    evaluateFourier,
    fitRidge,
    loadHeroLightcurves,
    loadRrLyraeCatalog,
    solveClocks,
    wesenheitIndex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / DEFAULT_CATALOG_CSV
LIGHTCURVES = REPO_ROOT / DEFAULT_LIGHTCURVE_CSV


class RrLyraeCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadRrLyraeCatalog(CATALOG)

    def test_the_catalogue_covers_both_clouds_and_both_modes(self) -> None:
        self.assertGreater(len(self.catalog), 35000)
        counts = self.catalog['cloud'].value_counts()
        self.assertGreater(int(counts['LMC']), 30000)
        self.assertGreater(int(counts['SMC']), 4000)
        types = set(self.catalog['type'])
        self.assertEqual(types, {'RRab', 'RRc'})

    def test_provenance_headers_are_present(self) -> None:
        header = ' '.join(
            line for line in CATALOG.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('ogle-iv', 'soszynski', 'j/aca/66/131', 'rrab', 'rrc', 'downloaded'):
            self.assertIn(token, header)
        self.assertIn('no extinction correction', header)
        photo = ' '.join(
            line for line in LIGHTCURVES.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('ogle-iv', 'i-band', '03686', 'downloaded'):
            self.assertIn(token, photo)

    def test_the_wesenheit_index_is_immune_to_reddening(self) -> None:
        visual, infrared = 19.0, 18.4
        reddening = 0.25
        extinguishedInfrared = infrared + WESENHEIT_COLOR_COEFFICIENT * reddening
        extinguishedVisual = visual + (WESENHEIT_COLOR_COEFFICIENT + 1.0) * reddening
        self.assertAlmostEqual(
            float(wesenheitIndex(np.array([visual]), np.array([infrared]))[0]),
            float(
                wesenheitIndex(np.array([extinguishedVisual]), np.array([extinguishedInfrared]))[0]
            ),
            places=9,
        )


class BaileyAndCandleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadRrLyraeCatalog(CATALOG)
        cls.solution = solveClocks(cls.catalog)

    def test_rrc_are_the_shorter_smaller_mode(self) -> None:
        """Bailey: first overtone is faster and a smaller bump."""
        lmc = self.catalog[self.catalog['cloud'] == 'LMC']
        ab = lmc[lmc['type'] == 'RRab']
        rc = lmc[lmc['type'] == 'RRc']
        self.assertLess(float(rc['period_days'].median()), float(ab['period_days'].median()))
        self.assertLess(self.solution.lmcRRcMedianAmplitude, self.solution.lmcRRabMedianAmplitude)
        self.assertGreater(
            self.solution.lmcRRabMedianAmplitude - self.solution.lmcRRcMedianAmplitude, 0.15
        )

    def test_wesenheit_tightens_the_rrab_ridge(self) -> None:
        self.assertGreater(self.solution.lmcInfrared.scatter, self.solution.lmcWesenheit.scatter)
        self.assertLess(self.solution.lmcWesenheit.scatter, 0.16)
        self.assertGreater(self.solution.lmcWesenheit.count / self.solution.lmcRRabCount, 0.85)

    def test_the_ridge_is_brighter_at_longer_period(self) -> None:
        self.assertLess(self.solution.lmcWesenheit.slope, 0.0)

    def test_the_smc_is_farther_but_short_of_the_eclipsing_binaries(self) -> None:
        """No metallicity term: RR Lyrae put the SMC too close."""
        self.assertGreater(self.solution.modulusOffset, 0.2)
        self.assertLess(self.solution.modulusOffset, 0.55)
        self.assertAlmostEqual(
            self.solution.distanceRatio, 10.0 ** (self.solution.modulusOffset / 5.0), places=9
        )
        self.assertLess(
            self.solution.smcDistanceKiloparsecs, self.solution.publishedSmcDistanceKiloparsecs
        )
        self.assertGreater(
            self.solution.publishedSmcDistanceKiloparsecs - self.solution.smcDistanceKiloparsecs,
            2.0,
        )

    def test_the_clip_does_not_throw_the_sample_away(self) -> None:
        lmc = self.catalog[(self.catalog['cloud'] == 'LMC') & (self.catalog['type'] == 'RRab')]
        unclipped = fitRidge(lmc['logPeriod'], lmc['wesenheit'], rounds=0)
        self.assertAlmostEqual(unclipped.slope, self.solution.lmcWesenheit.slope, delta=0.4)

    def test_the_distance_modulus_conversion_round_trips(self) -> None:
        self.assertAlmostEqual(distanceKiloparsecs(LMC_DISTANCE_MODULUS), 49.6, delta=0.2)
        self.assertAlmostEqual(distanceKiloparsecs(PUBLISHED_SMC_DISTANCE_MODULUS), 62.4, delta=0.5)


class HeroLightcurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadRrLyraeCatalog(CATALOG)
        cls.heroes = buildHeroCurves(cls.catalog, loadHeroLightcurves(LIGHTCURVES))

    def test_the_heroes_are_one_overtone_and_two_fundamentals(self) -> None:
        names = tuple(hero.star for hero in self.heroes)
        self.assertEqual(names, tuple(sorted(HERO_STARS, key=lambda star: self._period(star))))
        self.assertEqual([hero.subtype for hero in self.heroes], ['RRc', 'RRab', 'RRab'])
        self.assertLess(self.heroes[0].periodDays, self.heroes[1].periodDays)
        self.assertLess(self.heroes[1].periodDays, self.heroes[2].periodDays)

    def _period(self, star: str) -> float:
        return float(self.catalog.loc[self.catalog['star'] == star, 'period_days'].iloc[0])

    def test_baileys_law_is_visible_in_the_two_rrab(self) -> None:
        shortAb, longAb = self.heroes[1], self.heroes[2]
        self.assertGreater(shortAb.iAmplitude, longAb.iAmplitude)
        self.assertGreater(shortAb.iAmplitude - longAb.iAmplitude, 0.15)

    def test_folding_on_the_catalogue_period_is_coherent(self) -> None:
        for hero in self.heroes:
            fitted = evaluateFourier(hero.coefficients, hero.phase)
            residual = float(np.std(hero.magnitude - fitted))
            amplitude = float(np.ptp(evaluateFourier(hero.coefficients, np.linspace(0, 1, 256))))
            self.assertGreater(amplitude, 0.15)
            self.assertLess(residual, 0.25 * amplitude)
            self.assertGreaterEqual(len(hero.phase), 200)

    def test_the_fold_starts_at_maximum_light(self) -> None:
        for hero in self.heroes:
            phase = np.linspace(0.0, 1.0, 512, endpoint=False)
            brightest = float(phase[int(np.argmin(evaluateFourier(hero.coefficients, phase)))])
            self.assertLess(min(brightest, 1.0 - brightest), 0.03)


class RrLyraeAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.animator = RrLyraeCinematicAnimator(
            style='default', catalogCsvPath=CATALOG, lightcurveCsvPath=LIGHTCURVES
        )

    def test_every_act_is_reached_in_order(self) -> None:
        seen: list[str] = []
        for frame in range(ANIMATION_FRAMES):
            act = self.animator.act(frame)
            if not seen or seen[-1] != act:
                seen.append(act)
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['clouds'])

    def test_the_first_clock_is_on_screen_from_frame_zero(self) -> None:
        self.assertEqual(self.animator.heroReveal(0, 0), 1.0)
        self.assertEqual(self.animator.heroReveal(0, 1), 0.0)

    def test_each_playhead_runs_at_its_own_measured_rate(self) -> None:
        days = self.animator.elapsedDays(ANIMATION_FRAMES - 1)
        fast, _, slow = self.animator.heroes
        self.assertAlmostEqual(
            (days / fast.periodDays) / (days / slow.periodDays),
            slow.periodDays / fast.periodDays,
            places=9,
        )
        self.assertGreater(days / fast.periodDays, 6.0)
        self.assertGreater(days / slow.periodDays, 2.0)

    def test_the_captions_quote_the_numbers_that_were_fitted(self) -> None:
        solution = self.animator.solution
        bailey = self.animator.caption(ACT_BOUNDARIES[2][0] - 1)
        self.assertIn(f'{solution.lmcRRabCount}', bailey)
        self.assertIn(f'{solution.lmcRRcMedianAmplitude:.2f}', bailey)
        candle = self.animator.caption(ACT_BOUNDARIES[3][0] - 1)
        self.assertIn(f'{solution.lmcInfrared.scatter:.3f}', candle)
        self.assertIn(f'{solution.lmcWesenheit.scatter:.3f}', candle)
        payoff = self.animator.caption(ANIMATION_FRAMES - 1)
        self.assertIn(f'{solution.modulusOffset:.3f}', payoff)
        self.assertIn(f'{solution.smcDistanceKiloparsecs:.1f}', payoff)

    def test_the_live_fit_waits_until_the_plane_has_enough_stars(self) -> None:
        self.assertEqual(self.animator.candleReveal(0), 0.0)
        drawnAt = next(
            frame for frame in range(ANIMATION_FRAMES) if self.animator.fitReveal(frame) > 0.0
        )
        shown = int(round(self.animator.candleReveal(drawnAt) * len(self.animator.lmcRRabArrival)))
        self.assertGreaterEqual(shown, MINIMUM_POINTS_TO_FIT)


if __name__ == '__main__':
    unittest.main()
