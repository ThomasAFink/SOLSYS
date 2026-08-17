"""Tests for the pulsar lighthouse cinema (#103).

The film claims four things: that three real folded profiles have measured
periods and duty cycles, that each playhead runs on its own clock, that P-dot
follows from the catalogue's characteristic age, and that the Crab's clock is
the remnant. Each one is re-derived here from the committed CSVs, and the
captions are checked against those numbers.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.pulsar_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_CATALOG_CSV,
    DEFAULT_PROFILE_CSV,
    FILM_YEAR,
    HERO_JNAMES,
    SLOWDOWN,
    SN1054_YEAR,
    PulsarCinematicAnimator,
    buildHeroCurves,
    characteristicAgeYears,
    dutyCycle,
    formatAge,
    formatPeriod,
    impliedPeriodDerivative,
    loadHeroProfiles,
    loadPulsarCatalog,
    remnantAgeYears,
    secondaryPeakPhase,
    solveClocks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / DEFAULT_CATALOG_CSV
PROFILES = REPO_ROOT / DEFAULT_PROFILE_CSV


class PulsarCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadPulsarCatalog(CATALOG)

    def test_the_sample_is_the_atnf_catalogue(self) -> None:
        self.assertGreater(len(self.catalog), 1900)
        self.assertEqual(self.catalog['jname'].nunique(), len(self.catalog))
        self.assertTrue((self.catalog['period_s'] > 0.0).all())
        self.assertTrue((self.catalog['age_yr'] > 0.0).all())

    def test_the_three_heroes_are_in_the_catalogue(self) -> None:
        names = set(self.catalog['jname'])
        for jname in HERO_JNAMES:
            self.assertIn(jname, names)

    def test_provenance_headers_are_present(self) -> None:
        header = ' '.join(
            line for line in CATALOG.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('atnf', 'manchester', 'vizier', 'b/psr', 'downloaded', 'p/(2pdot)'):
            self.assertIn(token, header)
        photo = ' '.join(
            line for line in PROFILES.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('epn', 'gould', 'lyne', 'crab', 'johnston', 'downloaded', 'stokes i'):
            self.assertIn(token, photo)


class ClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadPulsarCatalog(CATALOG)
        cls.solution = solveClocks(cls.catalog)

    def test_characteristic_age_round_trips_through_pdot(self) -> None:
        crab = self.catalog[self.catalog['jname'] == 'J0534+2200'].iloc[0]
        pdot = impliedPeriodDerivative(float(crab['period_s']), float(crab['age_yr']))
        self.assertAlmostEqual(pdot, self.solution.crabPdot, places=20)
        self.assertAlmostEqual(
            characteristicAgeYears(float(crab['period_s']), pdot),
            float(crab['age_yr']),
            places=6,
        )

    def test_the_crab_pdot_is_the_canonical_spin_down(self) -> None:
        """Lyne, Pritchard & Smith 1988-ish: Crab P-dot is 4.2e-13 s/s."""
        self.assertGreater(self.solution.crabPdot, 4.0e-13)
        self.assertLess(self.solution.crabPdot, 4.5e-13)
        self.assertAlmostEqual(self.solution.crabAgeYr, 1260.0, places=1)

    def test_the_remnant_is_younger_than_the_characteristic_age(self) -> None:
        """τ = P/2Ṗ assumes birth at P = 0, so it overshoots the historical year."""
        self.assertEqual(self.solution.remnantAgeYr, remnantAgeYears())
        self.assertEqual(self.solution.remnantAgeYr, float(FILM_YEAR - SN1054_YEAR))
        self.assertGreater(self.solution.crabAgeYr, self.solution.remnantAgeYr)
        self.assertLess(self.solution.remnantAgeYr, 1000.0)
        self.assertGreater(self.solution.remnantAgeYr, 900.0)

    def test_the_crab_is_not_the_youngest_object_in_the_catalogue(self) -> None:
        """Magnetars beat it. The film has to say so."""
        self.assertGreater(self.solution.youngerThanCrab, 0)
        self.assertLess(self.solution.youngerThanCrab, 10)
        younger = self.catalog[self.catalog['age_yr'] < self.solution.crabAgeYr]
        # None of the younger objects is a 33 ms radio pulsar.
        self.assertTrue((younger['period_s'] > 0.2).all())

    def test_the_median_pulsar_is_millions_of_years_old(self) -> None:
        self.assertGreater(self.solution.medianAgeYr, 1.0e6)
        self.assertGreater(self.solution.sampleCount, 1900)


class HeroProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = loadPulsarCatalog(CATALOG)
        cls.heroes = buildHeroCurves(cls.catalog, loadHeroProfiles(PROFILES))

    def test_the_heroes_are_crab_vela_and_the_slow_clock(self) -> None:
        names = tuple(hero.displayName for hero in self.heroes)
        self.assertEqual(names, ('Crab', 'Vela', 'B0329+54'))
        periods = [hero.periodS for hero in self.heroes]
        self.assertLess(periods[0], periods[1])
        self.assertLess(periods[1], periods[2])
        self.assertAlmostEqual(periods[2] / periods[0], 21.4, delta=0.5)

    def test_duty_cycles_are_narrow(self) -> None:
        """A lighthouse beam, not a floodlight: W50 is a few percent of a turn."""
        for hero in self.heroes:
            self.assertGreater(hero.dutyCycle, 0.005)
            self.assertLess(hero.dutyCycle, 0.10)
            again = dutyCycle(
                loadHeroProfiles(PROFILES)[hero.jname]['stokes_i'].to_numpy(dtype=float)
            )
            self.assertAlmostEqual(again, hero.dutyCycle, places=9)

    def test_the_crab_has_an_interpulse(self) -> None:
        crab, vela, slow = self.heroes
        self.assertIsNotNone(crab.secondaryPhase)
        assert crab.secondaryPhase is not None
        separation = abs(crab.secondaryPhase - crab.peakPhase)
        self.assertGreater(min(separation, 1.0 - separation), 0.3)
        self.assertIsNone(vela.secondaryPhase)
        self.assertIsNone(slow.secondaryPhase)

    def test_the_epn_period_matches_the_atnf_clock(self) -> None:
        """The profile was folded at a 1989-ish epoch; ATNF P0 is later. Close."""
        for hero in self.heroes:
            self.assertGreater(len(hero.phase), 50)
            self.assertLess(abs(hero.profilePeriodS - hero.periodS) / hero.periodS, 0.01)
            self.assertTrue((hero.phase >= 0.0).all())
            self.assertTrue((hero.phase < 1.02).all())

    def test_playheads_run_at_each_pulsars_own_period(self) -> None:
        elapsed = 1.0
        crab, vela, slow = self.heroes
        self.assertAlmostEqual(crab.pulseCount(elapsed), elapsed / crab.periodS, places=9)
        self.assertGreater(crab.pulseCount(elapsed), 2 * vela.pulseCount(elapsed))
        self.assertGreater(vela.pulseCount(elapsed), 5 * slow.pulseCount(elapsed))
        self.assertAlmostEqual(crab.phaseAt(crab.periodS), crab.phaseAt(0.0), places=9)

    def test_formatters_match_the_captions(self) -> None:
        crab = self.heroes[0]
        self.assertEqual(formatPeriod(crab.periodS), f'{crab.periodS * 1000.0:.1f} ms')
        self.assertIn('yr', formatAge(1260.0))
        self.assertIn('Myr', formatAge(5.53e6))


class PulsarAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.animator = PulsarCinematicAnimator(
            style='default', catalogCsvPath=CATALOG, profileCsvPath=PROFILES
        )

    def test_every_act_is_reached_in_order(self) -> None:
        seen: list[str] = []
        for frame in range(ANIMATION_FRAMES):
            act = self.animator.act(frame)
            if not seen or seen[-1] != act:
                seen.append(act)
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['remnant'])

    def test_the_plane_opens_on_the_lighthouse_before_the_catalogue(self) -> None:
        self.assertEqual(self.animator.planeReveal(0), 0.0)
        beamFrame = ACT_BOUNDARIES[2][0] - 1
        self.assertGreater(self.animator.planeReveal(beamFrame), 0.9)
        self.assertEqual(self.animator.agesReveal(beamFrame), 0.0)
        self.assertGreater(self.animator.agesReveal(ANIMATION_FRAMES - 1), 0.9)

    def test_the_lighthouse_wedge_is_the_measured_duty_cycle(self) -> None:
        crab = self.animator.heroes[0]
        halfWidth = crab.dutyCycle * 360.0 / 2.0
        self.assertAlmostEqual(2.0 * halfWidth / 360.0, crab.dutyCycle, places=9)

    def test_the_live_census_lands_on_the_solved_catalogue(self) -> None:
        logP, logAge = self.animator.visibleCensus(ANIMATION_FRAMES - 1)
        self.assertEqual(len(logP), self.animator.solution.sampleCount)
        self.assertAlmostEqual(float(np.median(10.0**logAge)), self.animator.solution.medianAgeYr)

    def test_the_captions_quote_the_numbers_that_were_measured(self) -> None:
        solution = self.animator.solution
        crab = self.animator.heroes[0]
        pulseCaption = self.animator.caption(ACT_BOUNDARIES[0][0] - 1)
        self.assertIn(formatPeriod(crab.periodS), pulseCaption)
        self.assertIn(f'{100.0 * crab.dutyCycle:.1f}%', pulseCaption)
        trioCaption = self.animator.caption(ACT_BOUNDARIES[1][0] - 1)
        self.assertIn(f'×{SLOWDOWN:.0f}', trioCaption)
        agesCaption = self.animator.caption(ACT_BOUNDARIES[3][0] - 1)
        self.assertIn(f'{solution.sampleCount}', agesCaption)
        self.assertIn(f'{solution.crabPdot:.2e}', agesCaption)
        payoff = self.animator.caption(ANIMATION_FRAMES - 1)
        self.assertIn(formatAge(solution.crabAgeYr), payoff)
        self.assertIn(f'{solution.remnantAgeYr:.0f}', payoff)

    def test_the_title_counts_the_clocks_on_screen(self) -> None:
        self.assertIn('Crab', self.animator.title(0))
        self.assertIn('Three pulsars', self.animator.title(ACT_BOUNDARIES[1][0] - 1))

    def test_the_crab_is_on_screen_from_the_first_frame(self) -> None:
        self.assertEqual(self.animator.heroReveal(0, 0), 1.0)
        self.assertEqual(self.animator.heroReveal(0, 1), 0.0)

    def test_secondary_peak_helper_ignores_a_single_spike(self) -> None:
        phase = np.linspace(0.0, 1.0, 128, endpoint=False)
        spike = np.exp(-0.5 * ((phase - 0.4) / 0.02) ** 2)
        self.assertIsNone(secondaryPeakPhase(phase, spike))


if __name__ == '__main__':
    unittest.main()
