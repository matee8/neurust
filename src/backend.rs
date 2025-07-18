//! The generic computational backend.
//!
//! This module provides the [`Backend`] trait which defines the complete
//! contract for all linear algebra operations, data management, and tensor
//! creation.
//!
//! The default backend is [`ndarray`] and can be swapped out using crate
//! feature flags.

use num_traits::{One, Zero};

pub mod ndarray;

/// A trait that defines the contract for tensor operations that every
/// backend must fulfill.
///
/// The `Backend` trait is the core abstraction of this module. It provides a
/// generic interface for tensor creation, manipulation, and computation. By
/// implementing this trait, different computation libraries (like [`ndarray`]
/// or `nalgebra`) can be used as the underlying engine for tensor operations.
/// This allows for flexibility and performance tuning by switching backends
/// through feature flags. All functions are pure.
///
/// Some methods in this trait are marked `unsafe` because they do not perform
/// any invariant checks (e.g., for shape compatibility). The caller (typically
/// the [`TensorBase`](crate::tensor::TensorBase) wrapper) is responsible for
/// ensuring all preconditions are met before calling these functions.
pub trait Backend {
    /// The element type of the tensors.
    type Primitive: Clone + Zero + One;

    /// The concrete tensor representation provided by the backend.
    type Tensor;

    /// Adds two tensors element-wise.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `lhs` and `rhs` have the same shape.
    unsafe fn add(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Creates a tensor from a vector of data and a shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// -   The `shape` is valid (no zero dimensions, no overflow, see safety
    ///     notes on [`Backend::zeros()`]).
    /// -   The number of elements in `data` is equal to the product of the
    ///     `shape`'s dimensions.
    unsafe fn from_vec(
        data: Vec<Self::Primitive>,
        shape: &[usize],
    ) -> Self::Tensor;

    /// Returns the number of dimensions of the tensor.
    fn ndim(tensor: &Self::Tensor) -> usize;

    /// Creates a tensor with all elements set to one, with the given shape.
    ///
    /// # Safety
    ///
    /// See the safety notes for [`Backend::zeros()`].
    unsafe fn ones(shape: &[usize]) -> Self::Tensor;

    /// Returns the shape of the tensor as a slice of dimensions.
    fn shape(tensor: &Self::Tensor) -> &[usize];

    /// Creates a tensor with all elements set to zero, with the given shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `shape` slice does not contain any
    /// zeros, no dimensions overflow `isize`, and the product of axis lengths
    /// does not overflow [`isize::MAX`].
    unsafe fn zeros(shape: &[usize]) -> Self::Tensor;
}

cfg_if::cfg_if! {
    if #[cfg(feature = "ndarray-backend")] {
        /// Dynamically configured type alias for the selected backend, based
        /// on crate feature flags.
        pub type SelectedBackend<T> = ndarray::NdarrayBackend<T>;
    } else {
        compile_error!(
            "A backend feature must be enabled. Available: `backend-ndarray`"
        );
    }
}
