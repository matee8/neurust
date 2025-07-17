//! [`ndarray`] crate backend.

use core::marker::PhantomData;

use ndarray::{ArrayD, IxDyn};
use num_traits::{One, Zero};

use crate::backend::Backend;

/// Marker type for the [`ndarray`] backend.
#[derive(Debug)]
pub struct NdarrayBackend<T>
where
    T: Clone,
{
    _marker: PhantomData<T>,
}

impl<T> Backend for NdarrayBackend<T>
where
    T: Clone + Zero + One,
{
    type Tensor = ArrayD<T>;

    #[inline]
    fn ndim(tensor: &Self::Tensor) -> usize {
        tensor.ndim()
    }

    #[inline]
    unsafe fn ones(shape: &[usize]) -> Self::Tensor {
        ArrayD::ones(IxDyn(shape))
    }

    #[inline]
    fn shape(tensor: &Self::Tensor) -> &[usize] {
        tensor.shape()
    }

    #[inline]
    unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
        ArrayD::zeros(IxDyn(shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, ndarray::NdarrayBackend};

    #[test]
    fn ndarray_zeros_has_correct_shape() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(array.shape(), shape);
    }

    #[test]
    fn ndarray_zeros_has_correct_values() {
        let array = unsafe { NdarrayBackend::<f32>::zeros(&[2, 3]) };

        assert!(array.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn ndarray_ones_has_correct_shape() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::ones(shape) };

        assert_eq!(array.shape(), shape);
    }

    #[test]
    fn ndarray_ones_has_correct_values() {
        let array = unsafe { NdarrayBackend::<f32>::ones(&[2, 3]) };

        assert!(array.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn ndarray_shape_is_correct() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(NdarrayBackend::shape(&array), shape);
    }

    #[test]
    fn ndarray_ndim_is_correct() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(NdarrayBackend::ndim(&array), shape.len());
    }
}
