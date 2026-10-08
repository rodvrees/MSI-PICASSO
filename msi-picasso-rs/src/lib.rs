mod digest;
mod ion_image;
mod mob_coloc;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Batch compute monoisotopic mass and elemental composition for peptide sequences.
///
/// Returns tuple of 7 parallel lists:
///     (masses, mh_mzs, n_C, n_H, n_N, n_O, n_S)
#[pyfunction]
fn compute_peptide_masses(sequences: Vec<String>) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    digest::compute_peptide_info_batch(&sequences)
}

/// Match peptide [M+H]+ m/z values against MALDI feature m/z values.
///
/// Returns tuple of 3 parallel lists:
///     (feature_indices, peptide_indices, ppm_errors)
#[pyfunction]
#[pyo3(signature = (maldi_mzs, peptide_mzs, ppm_tolerance=20.0))]
fn match_mz(
    maldi_mzs: Vec<f64>,
    peptide_mzs: Vec<f64>,
    ppm_tolerance: f64,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    digest::match_mz_batch(&maldi_mzs, &peptide_mzs, ppm_tolerance)
}

/// Extract feature window integrals from a batch of profile-mode pixel spectra.
///
/// Processes pixels in parallel (rayon) using direct window summation
/// instead of the cumsum trick, which is faster when n_features × window_width
/// << n_mz (typical for MALDI with narrow extraction windows).
///
/// Args:
///     pixel_matrix: 2-D float32 numpy array of shape (n_pixels, n_mz).
///                   Must be C-contiguous (row-major).
///     lo_indices:   Inclusive start index for each feature window (list of int).
///     hi_indices:   Exclusive end index for each feature window (list of int).
///
/// Returns:
///     1-D float32 numpy array of length n_pixels * n_features (row-major).
///     Reshape to (n_pixels, n_features) in Python.
#[pyfunction]
fn accumulate_profile_chunk<'py>(
    py: Python<'py>,
    pixel_matrix: numpy::PyReadonlyArray2<'_, f32>,
    lo_indices: Vec<usize>,
    hi_indices: Vec<usize>,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
    use numpy::{IntoPyArray, PyUntypedArrayMethods};

    let shape = pixel_matrix.shape();
    let n_mz = shape[1];

    let mat_slice = pixel_matrix
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("pixel_matrix must be C-contiguous"))?;

    let mat_addr = mat_slice.as_ptr() as usize;
    let mat_len = mat_slice.len();

    let flat_output: Vec<f32> = py.allow_threads(move || {
        let mat = unsafe { std::slice::from_raw_parts(mat_addr as *const f32, mat_len) };
        ion_image::accumulate_profile_images(mat, n_mz, &lo_indices, &hi_indices)
    });

    Ok(numpy::ndarray::Array1::from_vec(flat_output)
        .into_pyarray(py)
        .into())
}

/// Compute per-candidate mobility-filtered colocalization and spatial features.
///
/// Processes all MALDI features in parallel (rayon).  The flat CSR arrays
/// must be built ONCE in Python from alphatims (49 500 calls) and reused here.
///
/// Args:
///     flat_mzs:            Concatenated sorted m/z arrays for all pixels (float32 numpy).
///     flat_scans:          Scan indices aligned with flat_mzs (uint32 numpy).
///     flat_ints:           Intensities aligned with flat_mzs (float32 numpy).
///     pixel_offsets:       CSR start offsets, length n_pixels + 1 (uint64 list).
///     pixel_xi:            X coordinates per pixel (uint32 list).
///     pixel_yi:            Y coordinates per pixel (uint32 list).
///     mob_values:          1/K0 value for each scan index (float64 list).
///     feature_mz_windows:  Flat float32 array, shape (n_features, 6, 2): [lo, hi] per offset.
///     cand_ptr:            CSR pointer over candidates per feature (uint32 list), len n_feat+1.
///     cand_k0_lo:          Lower 1/K0 bound per candidate (float64 list).
///     cand_k0_hi:          Upper 1/K0 bound per candidate (float64 list).
///     max_x:               Image width.
///     max_y:               Image height.
///
/// Returns:
///     1-D float32 numpy array of length n_total_cands * 10.
///     Reshape to (n_total_cands, 10) in Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mob_coloc_features<'py>(
    py: Python<'py>,
    flat_mzs: PyReadonlyArray1<'_, f32>,
    flat_scans: PyReadonlyArray1<'_, u32>,
    flat_ints: PyReadonlyArray1<'_, f32>,
    pixel_offsets: Vec<u64>,
    pixel_xi: Vec<u32>,
    pixel_yi: Vec<u32>,
    mob_values: Vec<f64>,
    feature_mz_windows: PyReadonlyArray1<'_, f32>,
    cand_ptr: Vec<u32>,
    cand_k0_lo: Vec<f64>,
    cand_k0_hi: Vec<f64>,
    max_x: usize,
    max_y: usize,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
    use numpy::IntoPyArray;

    let mzs_s = flat_mzs.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_mzs not C-contiguous"))?;
    let scans_s = flat_scans.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_scans not C-contiguous"))?;
    let ints_s = flat_ints.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_ints not C-contiguous"))?;
    let wins_s = feature_mz_windows.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("feature_mz_windows not C-contiguous"))?;

    // Cast to usize pointers so they cross the allow_threads (Ungil) boundary.
    // Safety: PyReadonlyArray1 holds a GIL-protected reference; arrays are
    // immutable while we hold the borrow; rayon threads only read.
    let mzs_addr = mzs_s.as_ptr() as usize;
    let mzs_len = mzs_s.len();
    let scans_addr = scans_s.as_ptr() as usize;
    let scans_len = scans_s.len();
    let ints_addr = ints_s.as_ptr() as usize;
    let ints_len = ints_s.len();
    let wins_addr = wins_s.as_ptr() as usize;
    let wins_len = wins_s.len();

    let flat_out: Vec<f32> = py.allow_threads(move || {
        let mzs = unsafe { std::slice::from_raw_parts(mzs_addr as *const f32, mzs_len) };
        let scans = unsafe { std::slice::from_raw_parts(scans_addr as *const u32, scans_len) };
        let ints = unsafe { std::slice::from_raw_parts(ints_addr as *const f32, ints_len) };
        let wins = unsafe { std::slice::from_raw_parts(wins_addr as *const f32, wins_len) };

        mob_coloc::compute_mob_coloc(
            mzs, scans, ints,
            &pixel_offsets, &pixel_xi, &pixel_yi,
            &mob_values,
            wins,
            &cand_ptr, &cand_k0_lo, &cand_k0_hi,
            max_x, max_y,
        )
    });

    Ok(numpy::ndarray::Array1::from_vec(flat_out)
        .into_pyarray(py)
        .into())
}

#[pymodule]
fn ms1rescore_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_peptide_masses, m)?)?;
    m.add_function(wrap_pyfunction!(match_mz, m)?)?;
    m.add_function(wrap_pyfunction!(accumulate_profile_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(mob_coloc_features, m)?)?;
    Ok(())
}
