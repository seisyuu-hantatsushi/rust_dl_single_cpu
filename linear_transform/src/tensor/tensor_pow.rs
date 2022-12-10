use num;

use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num+num::pow::Pow<T, Output = T>+Clone+Copy {

    pub fn pow_rank0(&self, i:T) -> Tensor<T> {
	assert_eq!(self.shape(), &[1,1]);
	Tensor::<T>::from_array(&[1,1], &[self[vec![0,0]].pow(i)])
    }

    pub fn pow(&self, i:T) -> Tensor<T> {
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| b.pow(i)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }

}
