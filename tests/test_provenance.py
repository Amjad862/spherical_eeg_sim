"""The recorded generator identity follows the code being executed."""

from spherical_eeg_sim import __version__
from spherical_eeg_sim import provenance


def test_imported_version_wins_over_stale_distribution(monkeypatch):
    installed_version = provenance.distribution_version

    def stale(name):
        return '1.0.0' if name == 'spherical_eeg_sim' else installed_version(name)

    monkeypatch.setattr(provenance, 'distribution_version', stale)
    versions = provenance.software_versions()
    assert versions['spherical_eeg_sim'] == __version__
    assert versions['spherical_eeg_sim_installed_distribution'] == '1.0.0'
    assert 'imported_package_path' not in provenance.environment_snapshot()
    assert provenance.environment_snapshot(include_paths=True)['imported_package_path']
