//! # Generic tensor module.
//!
//! This module provides a generic `Tensor` structure that is parameterized by
//! a backend. The `Backend` trait defines the complete contract for all
//! linear algebra operations, data management, and tensor creation.
//!
//! The default backend is `ndarray` and can be swapped out using crate
//! feature flags.

/// A trait that defines the contract for tensor operations that every
/// backend must fulfill.
///
/// The `Backend` trait is the core abstraction of the tensor module. It
/// provides a generic interface for tensor creation, manipulation, and
/// computation. By implementing this trait, different computation libraries
/// (like `ndarray` or `nalgebra`) can be used as the underlying engine for
/// tensor operations. This allows for flexibility and performance tuning
/// by switching backends through feature flags. All functions are pure.
pub trait Backend {
    /// The concrete tensor representation provided by the backend.
    type Tensor;

    /// Creates a tensor with all elements set to zero, with the given shape.
    fn zeros(shape: &[usize]) -> Self::Tensor;
}
