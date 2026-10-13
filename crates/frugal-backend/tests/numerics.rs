use frugal_backend::tensor::{checked_elements, Tensor};
use serde::Deserialize;

fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor::new(shape.to_vec(), values.to_vec()).unwrap()
}
fn close(actual: &[f32], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (f64::from(actual) - expected).abs();
        assert!(
            actual.is_finite() && error <= atol + rtol * expected.abs(),
            "actual={actual}, expected={expected}, error={error}"
        );
    }
}

#[test]
fn analytic_examples() {
    let a = tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let b = tensor(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[58., 64., 139., 154.]);
    for data in [[0., 0.], [1000., 1000.], [-1000., -1000.]] {
        assert_eq!(tensor(&[2], &data).softmax().unwrap().data(), &[0.5, 0.5]);
    }
    assert_eq!(tensor(&[1], &[0.]).silu().unwrap().data(), &[0.]);
    // RMS does not center the row, and applies each weight after normalizing.
    let normalized = tensor(&[2, 2], &[2., 2., 0., 0.])
        .rms_norm(&[1., -2.], 1.)
        .unwrap();
    close(
        normalized.data(),
        &[0.8944271909999159, -1.7888543819998317, 0., 0.],
        1e-6,
        2e-6,
    );
}

#[derive(Deserialize)]
struct Fixtures {
    absolute_tolerance: f64,
    relative_tolerance: f64,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    op: String,
    shape: Vec<usize>,
    data: Vec<f32>,
    expected: Vec<f64>,
    rhs_shape: Option<Vec<usize>>,
    rhs: Option<Vec<f32>>,
    weight: Option<Vec<f32>>,
    epsilon: Option<f32>,
}
#[test]
fn independent_decimal_fixtures() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("../../../evals/fixtures/numerics.json")).unwrap();
    for case in fixtures.cases {
        let input = Tensor::new(case.shape.clone(), case.data).unwrap();
        let (result, expected_shape) = match case.op.as_str() {
            "matmul" => {
                let rhs = Tensor::new(case.rhs_shape.unwrap(), case.rhs.unwrap()).unwrap();
                let shape = vec![case.shape[0], rhs.shape()[1]];
                (input.matmul(&rhs).unwrap(), shape)
            }
            "rms_norm" => (
                input
                    .rms_norm(&case.weight.unwrap(), case.epsilon.unwrap())
                    .unwrap(),
                case.shape,
            ),
            "softmax" => (input.softmax().unwrap(), case.shape),
            "silu" => (input.silu().unwrap(), case.shape),
            _ => panic!("unknown reference operation"),
        };
        assert_eq!(result.shape(), expected_shape);
        close(
            result.data(),
            &case.expected,
            fixtures.absolute_tolerance,
            fixtures.relative_tolerance,
        );
    }
}

#[test]
fn invalid_shapes_and_storage() {
    for shape in [vec![], vec![0], vec![2, 0], vec![0, 2], vec![1, 1, 1]] {
        assert!(Tensor::new(shape, vec![]).is_err());
    }
    assert!(Tensor::new(vec![2, 3], vec![0.; 5]).is_err());
    assert!(Tensor::new(vec![1], vec![0., 1.]).is_err());
    assert!(tensor(&[2, 3], &[0.; 6])
        .matmul(&tensor(&[2, 2], &[0.; 4]))
        .is_err());
    assert!(tensor(&[2], &[1., 2.])
        .matmul(&tensor(&[2, 1], &[1., 2.]))
        .is_err());
    assert!(tensor(&[2, 1], &[1., 2.])
        .matmul(&tensor(&[2], &[1., 2.]))
        .is_err());
}

#[test]
fn shape_and_allocation_overflow_without_allocating() {
    assert!(checked_elements(&[usize::MAX, 2])
        .unwrap_err()
        .to_string()
        .contains("shape multiplication"));
    assert!(checked_elements(&[usize::MAX / 4 + 1])
        .unwrap_err()
        .to_string()
        .contains("byte-size overflow"));
    assert!(checked_elements(&[isize::MAX as usize / 4 + 1])
        .unwrap_err()
        .to_string()
        .contains("isize::MAX"));
    assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
}

#[test]
fn invalid_parameters_and_nonfinite_inputs() {
    let x = tensor(&[2, 2], &[1., 2., 3., 4.]);
    for epsilon in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(x.rms_norm(&[1., 1.], epsilon).is_err());
    }
    assert!(x.rms_norm(&[], 1e-6).is_err());
    assert!(x.rms_norm(&[1.], 1e-6).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Every operation shares the immutable constructor's finite-input invariant.
        assert!(Tensor::new(vec![1], vec![value]).is_err());
        assert!(x.rms_norm(&[value, 1.], 1e-6).is_err());
    }
}

#[test]
fn extreme_finite_inputs_and_output_overflow() {
    let x = tensor(&[2], &[f32::MAX, -f32::MAX]);
    assert_eq!(x.softmax().unwrap().data(), &[1., 0.]);
    assert_eq!(x.silu().unwrap().data(), &[f32::MAX, -0.]);
    close(
        x.rms_norm(&[1., 1.], f32::MIN_POSITIVE).unwrap().data(),
        &[1., -1.],
        1e-6,
        2e-6,
    );
    assert!(tensor(&[1, 1], &[f32::MAX])
        .matmul(&tensor(&[1, 1], &[2.]))
        .is_err());
    assert!(tensor(&[2], &[1., 0.])
        .rms_norm(&[f32::MAX, 1.], 1e-6)
        .is_err());
}

#[test]
fn row_axis_shift_invariance_and_probability_sum() {
    let original = tensor(&[2, 3], &[-2., 0., 2., 10., 11., 12.]);
    let shifted = tensor(&[2, 3], &[998., 1000., 1002., -990., -989., -988.]);
    let probabilities = original.softmax().unwrap();
    close(
        shifted.softmax().unwrap().data(),
        &probabilities
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        1e-6,
        2e-6,
    );
    for row in probabilities.data().chunks_exact(3) {
        assert!((row.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert!(row.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
    assert_eq!(
        tensor(&[2, 1], &[-1000., 1000.]).softmax().unwrap().data(),
        &[1., 1.]
    );
}
