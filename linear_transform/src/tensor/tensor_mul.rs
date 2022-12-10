use num;
use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn mul_rank0(lhs:&Tensor<T>, rhs:&Tensor<T>) -> Tensor<T> {
	assert_eq!(lhs.shape(), &[1,1]);
	assert_eq!(rhs.shape(), &[1,1]);
	Tensor::<T>::from_array(&[1,1], &[lhs[vec![0,0]]*rhs[vec![0,0]]])
    }

    pub fn hadamard_product(lhs:&Tensor<T>, rhs:&Tensor<T>) -> Tensor<T> {
	assert_eq!(lhs.shape(), rhs.shape());
	let v = lhs.buffer().iter().zip(rhs.buffer().iter()).map(|(&l, &r)| { l*r }).collect::<Vec<T>>();
	Tensor::from_vector(lhs.shape().to_vec(), v)
    }
}
