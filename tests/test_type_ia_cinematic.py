"""Tests for the Type Ia standard-candle cinema (#126).

The film claims three things: that a real B-band light curve fades by a
measured Δm15, that slower decliners are brighter (Phillips), and that after
standardization the Hubble diagram has slope 5. All three are re-derived here
from the committed CSVs.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.type_ia_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_CATALOG_CSV,
    DEFAULT_LIGHTCURVE_CSV,
    HERO_NAMES,
    HUBBLE_FLOW_Z,
    INVERSE_SQUARE_SLOPE,
    MINIMUM_POINTS_TO_FIT,
    TypeIaCandleCinematicAnimator,
    buildHeroCurves,
    declineFifteen,
    geometricModulus,
    loadHeroLightcurves,
    loadTypeIaCatalog,
    solveCandle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / DEFAULT_CATALOG_CSV
LIGHTCURVES = REPO_ROOT / DEFAULT_LIGHTCURVE_CSV


class TypeIaCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadTypeIaCatalog(CATALOG)

    def test_the_sample_is_the_pantheonplus_release(self) -> None:
        self.assertGreater(len(self.catalog), 1400)
        self.assertEqual(self.catalog['name'].nunique(), len(self.catalog))
        self.assertGreater(self.catalog['z_hd'].max(), 2.0)
        self.assertLess(self.catalog['z_hd'].min(), 0.002)

    def test_cepheid_calibrators_are_marked(self) -> None:
        calibrators = self.catalog[self.catalog['is_calibrator'] == 1]
        self.assertGreater(len(calibrators), 30)
        self.assertTrue(calibrators['cepheid_mu'].notna().all())
        self.assertTrue((calibrators['cepheid_mu'] > 20.0).all())
        self.assertTrue(
            self.catalog.loc[self.catalog['is_calibrator'] == 0, 'cepheid_mu'].isna().all()
        )

    def test_provenance_headers_are_present(self) -> None:
        header = ' '.join(
            line for line in CATALOG.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('pantheon+', 'scolnic', 'sh0es', 'downloaded', 'one row per'):
            self.assertIn(token, header)
        photo = ' '.join(
            line for line in LIGHTCURVES.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('open supernova catalog', 'b-band', '2011fe', 'downloaded'):
            self.assertIn(token, photo)


class PhillipsAndHubbleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadTypeIaCatalog(CATALOG)
        cls.solution = solveCandle(cls.catalog)

    def test_the_hubble_slope_is_the_inverse_square_law(self) -> None:
        self.assertAlmostEqual(self.solution.hubble.slope, INVERSE_SQUARE_SLOPE, delta=0.4)
        self.assertGreater(self.solution.hubble.count, 300)
        self.assertLess(self.solution.hubble.scatter, 0.25)

    def test_slower_stretch_is_brighter(self) -> None:
        """Phillips: larger SALT2 x1 (slower) means a brighter peak."""
        self.assertLess(self.solution.phillips.slope, 0.0)
        self.assertGreater(abs(self.solution.phillips.slope), 0.08)
        self.assertLess(abs(self.solution.phillips.slope), 0.35)

    def test_the_hubble_flow_window_is_populated(self) -> None:
        flow = self.catalog[self.catalog['z_hd'].between(*HUBBLE_FLOW_Z)]
        self.assertEqual(len(flow), self.solution.hubbleFlowCount)
        self.assertGreater(self.solution.hubbleFlowCount, 200)


class HeroLightcurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadTypeIaCatalog(CATALOG)
        cls.heroes = buildHeroCurves(cls.catalog, loadHeroLightcurves(LIGHTCURVES))

    def test_the_heroes_span_stretch(self) -> None:
        names = tuple(hero.name for hero in self.heroes)
        self.assertEqual(names, HERO_NAMES)
        declines = [hero.decline15 for hero in self.heroes]
        # 2000cn fast, 2011fe normal, 2005eq slow.
        self.assertGreater(declines[1], declines[0])
        self.assertGreater(declines[0], declines[2])
        self.assertGreater(declines[1] - declines[2], 0.4)

    def test_the_slow_decliner_is_the_bright_one(self) -> None:
        fast, slow = self.heroes[1], self.heroes[2]
        self.assertLess(slow.absolutePeak, fast.absolutePeak)
        self.assertGreater(fast.absolutePeak - slow.absolutePeak, 0.3)

    def test_decline_is_measured_from_the_photometry(self) -> None:
        for hero in self.heroes:
            again = declineFifteen(hero.daysFromPeak, hero.magnitude)
            self.assertAlmostEqual(again, hero.decline15, places=9)
            self.assertGreater(hero.decline15, 0.6)
            self.assertLess(hero.decline15, 2.2)
            self.assertGreaterEqual(len(hero.daysFromPeak), 20)

    def test_2011fe_sits_on_the_cepheid_rung(self) -> None:
        hero = self.heroes[0]
        row = self.catalog[self.catalog['name'] == '2011fe'].iloc[0]
        self.assertEqual(int(row['is_calibrator']), 1)
        self.assertAlmostEqual(hero.modulus, float(row['cepheid_mu']), places=6)
        self.assertNotEqual(
            hero.modulus,
            geometricModulus(hero.redshift, float('nan')),
        )


class CandleAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.animator = TypeIaCandleCinematicAnimator(
            style='default', catalogCsvPath=CATALOG, lightcurveCsvPath=LIGHTCURVES
        )

    def test_every_act_is_reached_in_order(self) -> None:
        seen: list[str] = []
        for frame in range(ANIMATION_FRAMES):
            act = self.animator.act(frame)
            if not seen or seen[-1] != act:
                seen.append(act)
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['ruler'])

    def test_stretching_time_collapses_the_declines(self) -> None:
        """At morph=1 the drawn widths share 2011fe's Δm15."""
        start = self.animator.stretchMorph(0)
        end = self.animator.stretchMorph(ANIMATION_FRAMES - 1)
        self.assertEqual(start, 0.0)
        self.assertEqual(end, 1.0)
        reference = self.animator.heroes[0]
        fast = self.animator.heroes[1]
        unstretched = declineFifteen(fast.daysFromPeak, fast.magnitude)
        stretched = self.animator.stretchedDays(fast, ANIMATION_FRAMES - 1)
        drawnDecline = declineFifteen(stretched, fast.magnitude)
        self.assertLess(
            abs(drawnDecline - reference.decline15), abs(unstretched - reference.decline15) * 0.5
        )

    def test_the_live_hubble_fit_lands_on_the_solved_relation(self) -> None:
        logZ, mu = self.animator.visibleHubble(ANIMATION_FRAMES - 1)
        live = self.animator.liveHubble(logZ, mu)
        self.assertIsNotNone(live)
        assert live is not None
        self.assertAlmostEqual(live.slope, self.animator.solution.hubble.slope, places=9)

    def test_the_plane_fills_before_it_is_fitted(self) -> None:
        self.assertEqual(self.animator.planeReveal(0), 0.0)
        drawnAt = next(
            frame for frame in range(ANIMATION_FRAMES) if self.animator.fitReveal(frame) > 0.0
        )
        logZ, _ = self.animator.visibleHubble(drawnAt)
        inFlow = (logZ >= np.log10(HUBBLE_FLOW_Z[0])) & (logZ <= np.log10(HUBBLE_FLOW_Z[1]))
        self.assertGreaterEqual(int(inFlow.sum()), MINIMUM_POINTS_TO_FIT)

    def test_the_captions_quote_the_numbers_that_were_fitted(self) -> None:
        solution = self.animator.solution
        stretchCaption = self.animator.caption(ACT_BOUNDARIES[2][0] - 1)
        self.assertIn(f'{solution.phillips.slope:+.2f}', stretchCaption)
        hubbleCaption = self.animator.caption(ACT_BOUNDARIES[3][0] - 1)
        self.assertIn(f'{solution.hubble.slope:.2f}', hubbleCaption)
        payoff = self.animator.caption(ANIMATION_FRAMES - 1)
        self.assertIn(f'{solution.calibratorCount}', payoff)

    def test_the_title_counts_the_blasts_on_screen(self) -> None:
        self.assertIn('SN 2011fe', self.animator.title(0))
        self.assertIn('Three Type Ia', self.animator.title(ANIMATION_FRAMES - 1))


if __name__ == '__main__':
    unittest.main()
