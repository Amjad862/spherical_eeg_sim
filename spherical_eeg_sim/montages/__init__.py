"""Bundled electrode montage CSVs.

CSV files in this directory are resolved by
``spherical_eeg_sim.core.electrodes.resolve_electrode_path``. Users may pass
either a filesystem path to a custom CSV or a bundled-montage basename such as
``'standard_1010_69.csv'``.

All positions lie on the model scalp sphere and are placed there by radial
projection of vetted, BSD-licensed reference templates (see ``_templates/`` and
``_generate.py`` for provenance, attribution, and the reproducible generator).
When a montage is loaded through the pipeline, its positions are rescaled to the
configured scalp radius (``head.radii_m[-1]``), so a montage is not tied to the
radius at which its CSV happens to be stored.

Standard 10-20 / 10-10 / 10-05 family
  (Oostenveld & Praamstra, 2001; Jurcak et al., 2007)
  - ``standard_1020_19.csv``   classic clinical 10-20, 19 channels
  - ``standard_1020_20.csv``   10-20 + midline, 20 channels
  - ``standard_1010_32.csv``   10-10 subset, 32 channels
  - ``standard_1010_61.csv``   10-10, 61 channels
  - ``standard_1010_69.csv``   full 10-10, 69 channels
  - ``standard_1005_336.csv``  full 10-05 high-density, 336 channels

Vendor nets -- spherical approximations (``_spherical`` suffix)
  Projected from the manufacturers' published layouts; adapted to the sphere,
  not the original 3-D geometry.
  - ``biosemi_16/32/64/128/256_spherical.csv``          BioSemi caps
  - ``gsn_hydrocel_32/64/128/256/257_spherical.csv``    EGI/Philips Geodesic
                                                        Sensor Net
"""
