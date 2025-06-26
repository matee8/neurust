//! # The tensor module
//!
//! This module provides the core data structure for representing
//! multi-dimensional arrays, which are fundamental to all numerical
//! computations in the library.

use ndarray::ArrayD;

/// Creates a `Tensor` from nested arrays or vectors with a `vec!`-like syntax.
/// The data type of the tensor's elements is inferred from the literals.
///
/// # Examples
///
/// ```
/// use neurust::tensor;
///
/// // A 1D Tensor
/// let v = tensor![1.0, 2.0, 3.0];
///
/// // A 2D Tensor
/// let m = tensor![[1.0, 2.0], [3.0, 4.0]];
///
/// // The macro also works with other numeric types like integers.
/// let i = tensor![1, 2, 3];
/// ```
#[macro_export]
macro_rules! tensor {
    ($($data:tt)+) => {
        $crate::tensor::Tensor::from(ndarray::array!($($data)+).into_dyn())
    };
}

/// A generic, multi-dimensional array holding elements of type `T`.
///
/// A `Tensor` represents a grid of elements with a specific shape. Unlike
/// statically-sized arrays, the number of dimensions (or rank) of a `Tensor`
/// is determined at runtime, providing flexibility for numerical computations.
///
/// The element type `T` is typically a numeric type (e.g., `f32`, `i64`).
#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<T> {
    inner: ArrayD<T>,
}

/// Provides interoperability with the `ndarray` crate.
///
/// Enables a direct and efficient conversion from `ndarray`'s
/// dynamically-dimensioned array type into a `Tensor`.
impl<T> From<ArrayD<T>> for Tensor<T> {
    #[inline]
    fn from(value: ArrayD<T>) -> Self {
        Self { inner: value }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_tensor_macro_1d() {
        let tensor = tensor![1.0, 2.0];
        let expected = Tensor::from(ndarray::arr1(&[1.0, 2.0]).into_dyn());

        assert_eq!(tensor, expected)
    }

    #[test]
    fn test_tensor_macro_2d() {
        let tensor = tensor![[1.0, 2.0], [3.0, 4.0]];
        let expected =
            Tensor::from(ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());

        assert_eq!(tensor, expected)
    }

    #[test]
    fn test_tensor_macro_3d() {
        let tensor =
            tensor![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let expected = Tensor::from(
            ndarray::arr3(&[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ])
            .into_dyn(),
        );

        assert_eq!(tensor, expected)
    }
}
