"""Tests for the Cepheid period–luminosity cinema (#159).

The film makes four claims: that three real light curves fold coherently on their
catalogued periods, that the slower Cepheids are the brighter ones, that the
Wesenheit combination cancels reddening rather than merely flattering the plot,
and that the gap between the two clouds' ridges is a distance. Each one is
re-derived here from the committed CSVs, and the captions are checked against the
fits so the prose cannot drift away from the numbers.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.cepheid_ladder_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_CATALOG_CSV,
    DEFAULT_LIGHTCURVE_CSV,
    FIT_PERIOD_RANGE_DAYS,
    HERO_STARS,
    LMC_DISTANCE_MODULUS,
    MINIMUM_POINTS_TO_FIT,
    PUBLISHED_LMC_WESENHEIT_SLOPE,
    PUBLISHED_SMC_DISTANCE_MODULUS,
    WESENHEIT_COLOR_COEFFICIENT,
    CepheidLadderCinematicAnimator,
    buildHeroCurves,
    distanceKiloparsecs,
    evaluateFourier,
    fitRidge,
    loadCepheidCatalog,
    loadHeroLightcurves,
    solveLadder,
    wesenheitIndex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / DEFAULT_CATALOG_CSV
LIGHTCURVES = REPO_ROOT / DEFAULT_LIGHTCURVE_CSV


class CepheidCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadCepheidCatalog(CATALOG)

    def test_the_catalogue_covers_both_clouds(self) -> None:
        counts = self.catalog['cloud'].value_counts()
        self.assertGreater(int(counts['LMC']), 2000)
        self.assertGreater(int(counts['SMC']), 2000)
        self.assertEqual(set(counts.index), {'LMC', 'SMC'})
        self.assertFalse(
            self.catalog[['period_days', 'mean_v_mag', 'mean_i_mag']].isna().any().any()
        )

    def test_the_sample_is_the_classical_instability_strip(self) -> None:
        periods = self.catalog['period_days'].to_numpy()
        self.assertGreaterEqual(periods.min(), FIT_PERIOD_RANGE_DAYS[0])
        self.assertLessEqual(periods.max(), FIT_PERIOD_RANGE_DAYS[1])
        # Cepheids are cool enough to be redder in V - I than any plausible error.
        colour = self.catalog['mean_v_mag'] - self.catalog['mean_i_mag']
        self.assertGreater(float(colour.min()), 0.0)
        self.assertLess(float(colour.median()), 1.5)

    def test_the_wesenheit_index_is_immune_to_reddening(self) -> None:
        """Redden a star and the index must not move — that is the whole point.

        Extinction adds A_I to I and E(V - I) to the colour, with
        A_I = 1.55 E(V - I) in these bands, so the combination cancels exactly.
        """
        visual, infrared = 15.4, 14.6
        reddening = 0.35
        extinguishedInfrared = infrared + WESENHEIT_COLOR_COEFFICIENT * reddening
        extinguishedVisual = visual + (WESENHEIT_COLOR_COEFFICIENT + 1.0) * reddening
        self.assertAlmostEqual(
            float(wesenheitIndex(np.array([visual]), np.array([infrared]))[0]),
            float(
                wesenheitIndex(np.array([extinguishedVisual]), np.array([extinguishedInfrared]))[0]
            ),
            places=9,
        )

    def test_provenance_headers_are_present(self) -> None:
        catalogHeader = ' '.join(
            line for line in CATALOG.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('ogle-iv', 'soszynski', 'j/aca/65/297', 'fundamental', 'downloaded'):
            self.assertIn(token, catalogHeader)
        # The film corrects for reddening on screen, so the stored magnitudes
        # have to be advertised as uncorrected.
        self.assertIn('no extinction correction', catalogHeader)

        photometryHeader = ' '.join(
            line for line in LIGHTCURVES.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('gaia dr3', 'epoch photometry', 'transit', 'downloaded'):
            self.assertIn(token, photometryHeader)


class LeavittLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadCepheidCatalog(CATALOG)
        cls.solution = solveLadder(cls.catalog)

    def test_the_lmc_slope_reproduces_the_published_fit(self) -> None:
        slope = self.solution.lmcWesenheit.slope
        self.assertAlmostEqual(slope, PUBLISHED_LMC_WESENHEIT_SLOPE, delta=0.05)
        self.assertGreater(self.solution.lmcWesenheit.count, 2000)

    def test_removing_reddening_tightens_the_ridge(self) -> None:
        visual = self.solution.lmcVisual.scatter
        infrared = self.solution.lmcInfrared.scatter
        wesenheit = self.solution.lmcWesenheit.scatter
        self.assertGreater(visual, infrared)
        self.assertGreater(infrared, wesenheit)
        # Under a tenth of a magnitude is what makes a Cepheid a distance
        # indicator rather than a curiosity.
        self.assertLess(wesenheit, 0.10)

    def test_the_clip_rejects_only_the_stragglers(self) -> None:
        """The fit must not owe its tightness to throwing most of the sample away."""
        lmc = self.catalog[self.catalog['cloud'] == 'LMC']
        kept = self.solution.lmcWesenheit.count
        self.assertGreater(kept / len(lmc), 0.93)
        unclipped = fitRidge(lmc['logPeriod'], lmc['wesenheit'], rounds=0)
        self.assertAlmostEqual(unclipped.slope, self.solution.lmcWesenheit.slope, delta=0.1)

    def test_a_steeper_relation_needs_longer_periods_to_be_brighter(self) -> None:
        """Sanity of sign: the ridge must rise in brightness with period."""
        self.assertLess(self.solution.lmcWesenheit.slope, 0.0)
        tenDay = self.solution.lmcWesenheit.zeroPoint
        hundredDay = self.solution.lmcWesenheit.slope * 1.0 + tenDay
        self.assertLess(hundredDay, tenDay)


class CloudDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solution = solveLadder(loadCepheidCatalog(CATALOG))

    def test_the_two_ridges_are_parallel(self) -> None:
        difference = abs(self.solution.smcWesenheit.slope - self.solution.lmcWesenheit.slope)
        self.assertLess(difference, 0.25)

    def test_the_offset_is_a_distance_ratio(self) -> None:
        offset = self.solution.modulusOffset
        self.assertGreater(offset, 0.0)
        self.assertAlmostEqual(self.solution.distanceRatio, 10.0 ** (offset / 5.0), places=9)
        self.assertGreater(
            self.solution.smcDistanceKiloparsecs, self.solution.lmcDistanceKiloparsecs
        )

    def test_the_derived_smc_distance_agrees_with_eclipsing_binaries(self) -> None:
        """The payoff: a Cepheid distance checked against a geometric one."""
        self.assertAlmostEqual(
            self.solution.smcDistanceModulus, PUBLISHED_SMC_DISTANCE_MODULUS, delta=0.10
        )
        self.assertAlmostEqual(
            self.solution.smcDistanceKiloparsecs,
            self.solution.publishedSmcDistanceKiloparsecs,
            delta=3.0,
        )

    def test_the_distance_modulus_conversion_round_trips(self) -> None:
        self.assertAlmostEqual(distanceKiloparsecs(LMC_DISTANCE_MODULUS), 49.6, delta=0.2)
        self.assertAlmostEqual(distanceKiloparsecs(0.0), 0.01, places=6)
        recovered = 5.0 * np.log10(distanceKiloparsecs(18.5) * 1000.0) - 5.0
        self.assertAlmostEqual(recovered, 18.5, places=9)


class HeroLightcurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadCepheidCatalog(CATALOG)
        cls.heroes = buildHeroCurves(cls.catalog, loadHeroLightcurves(LIGHTCURVES))

    def test_the_heroes_span_a_decade_of_period(self) -> None:
        periods = [hero.periodDays for hero in self.heroes]
        self.assertEqual(len(periods), len(HERO_STARS))
        self.assertEqual(periods, sorted(periods))
        self.assertGreater(periods[-1] / periods[0], 10.0)

    def test_the_slow_hero_is_the_bright_one(self) -> None:
        """Leavitt's law visible in three stars, before any fitting."""
        magnitudes = [hero.meanInfraredMag for hero in self.heroes]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        self.assertGreater(magnitudes[0] - magnitudes[-1], 2.0)

    def test_folding_on_the_catalogue_period_is_coherent(self) -> None:
        """If the period were wrong the fold would smear into a band, not a curve."""
        for hero in self.heroes:
            fitted = evaluateFourier(hero.coefficients, hero.phase)
            residual = float(np.std(hero.magnitude - fitted))
            amplitude = float(np.ptp(evaluateFourier(hero.coefficients, np.linspace(0, 1, 256))))
            self.assertGreater(amplitude, 0.3)
            self.assertLess(residual, 0.2 * amplitude)

    def test_the_fold_starts_at_maximum_light(self) -> None:
        for hero in self.heroes:
            phase = np.linspace(0.0, 1.0, 512, endpoint=False)
            brightest = float(phase[int(np.argmin(evaluateFourier(hero.coefficients, phase)))])
            self.assertLess(min(brightest, 1.0 - brightest), 0.03)

    def test_the_phase_coverage_leaves_no_large_gap(self) -> None:
        for hero in self.heroes:
            occupied = np.histogram(hero.phase, bins=10, range=(0.0, 1.0))[0]
            self.assertGreaterEqual(int((occupied > 0).sum()), 8)
            self.assertGreaterEqual(len(hero.phase), 30)


class LadderAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.animator = CepheidLadderCinematicAnimator(
            style='default', catalogCsvPath=CATALOG, lightcurveCsvPath=LIGHTCURVES
        )

    def test_every_act_is_reached_in_order(self) -> None:
        seen: list[str] = []
        for frame in range(ANIMATION_FRAMES):
            act = self.animator.act(frame)
            if not seen or seen[-1] != act:
                seen.append(act)
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['clouds'])

    def test_each_playhead_runs_at_its_own_measured_rate(self) -> None:
        """The pulse panel's whole argument: same film time, different clocks."""
        days = self.animator.elapsedDays(ANIMATION_FRAMES - 1)
        fast, _, slow = self.animator.heroes
        self.assertAlmostEqual(
            (days / fast.periodDays) / (days / slow.periodDays),
            slow.periodDays / fast.periodDays,
            places=9,
        )
        self.assertGreater(days / fast.periodDays, 10.0)
        self.assertGreater(days / slow.periodDays, 1.0)

    def test_the_reddening_morph_runs_from_mean_i_to_the_wesenheit_index(self) -> None:
        infrared = self.animator.lmcInfrared
        wesenheit = self.animator.lmcWesenheit
        start = self.animator.morphedMagnitude(infrared, wesenheit, 0.0)
        end = self.animator.morphedMagnitude(infrared, wesenheit, 1.0)
        np.testing.assert_allclose(start, infrared)
        np.testing.assert_allclose(end, wesenheit)
        self.assertEqual(self.animator.wesenheitMorph(0), 0.0)
        self.assertEqual(self.animator.wesenheitMorph(ANIMATION_FRAMES - 1), 1.0)

    def test_the_plane_fills_before_it_is_fitted(self) -> None:
        first = self.animator.act(0)
        self.assertEqual(first, ACT_BOUNDARIES[0][1])
        self.assertEqual(self.animator.planeReveal(0), 0.0)
        drawnAt = next(
            frame for frame in range(ANIMATION_FRAMES) if self.animator.fitReveal(frame) > 0.0
        )
        logPeriod, _ = self.animator.visibleLmc(drawnAt)
        self.assertGreaterEqual(len(logPeriod), MINIMUM_POINTS_TO_FIT)

    def test_the_live_fit_lands_on_the_solved_relation(self) -> None:
        """What the last frame prints has to be what the module measured."""
        logPeriod, magnitude = self.animator.visibleLmc(ANIMATION_FRAMES - 1)
        live = self.animator.liveFit(logPeriod, magnitude)
        self.assertIsNotNone(live)
        assert live is not None
        self.assertAlmostEqual(live.slope, self.animator.solution.lmcWesenheit.slope, places=9)
        self.assertAlmostEqual(live.scatter, self.animator.solution.lmcWesenheit.scatter, places=9)

    def test_the_window_holds_the_points_it_draws(self) -> None:
        """Framing is derived, so the heroes cannot slide off the top of the plot."""
        frame = ANIMATION_FRAMES - 1
        faint, bright = self.animator.planeLimits(frame)
        self.assertGreater(faint, bright)
        for cloud in (self.animator.visibleLmc(frame), self.animator.visibleSmc(frame)):
            _, magnitude = cloud
            inside = (magnitude <= faint) & (magnitude >= bright)
            self.assertGreater(inside.mean(), 0.99)
        morph = self.animator.wesenheitMorph(frame)
        for hero in self.animator.heroes:
            placed = hero.meanInfraredMag + morph * (hero.wesenheitMag - hero.meanInfraredMag)
            self.assertLess(placed, faint)
            self.assertGreater(placed, bright)

    def test_the_captions_quote_the_numbers_that_were_fitted(self) -> None:
        solution = self.animator.solution
        wesenheitCaption = self.animator.caption(ACT_BOUNDARIES[3][0] - 1)
        self.assertIn(f'{solution.lmcInfrared.scatter:.3f}', wesenheitCaption)
        self.assertIn(f'{solution.lmcWesenheit.scatter:.3f}', wesenheitCaption)

        payoff = self.animator.caption(ANIMATION_FRAMES - 1)
        self.assertIn(f'{solution.modulusOffset:.3f}', payoff)
        self.assertIn(f'{solution.distanceRatio:.2f}', payoff)
        self.assertIn(f'{solution.smcDistanceKiloparsecs:.0f}', payoff)

    def test_the_title_counts_the_stars_actually_on_screen(self) -> None:
        self.assertIn('A Cepheid', self.animator.title(0))
        self.assertIn('Three Cepheids', self.animator.title(ANIMATION_FRAMES - 1))


if __name__ == '__main__':
    unittest.main()
