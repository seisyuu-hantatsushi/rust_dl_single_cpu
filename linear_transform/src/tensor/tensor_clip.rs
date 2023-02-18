/* -*- tab-width:4 -*- */
use num;

use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy+std::cmp::PartialOrd {

    pub fn clip(&self, min:T, max:T) -> Tensor<T> {
		assert!(min < max);
		let cliped_v = self.buffer().iter()
			.map(|&e| { if e > max { max } else if e < min { min } else { e } } ).collect::<Vec<T>>();
		Tensor::from_vector(self.shape().to_vec(), cliped_v)
    }

    pub fn clip_self(&mut self, min:T, max:T) {
		assert!(min < max);
		let clip_v = self.buffer().iter()
			.map(|&e| { if e > max { max } else if e < min { min } else { e }  } ).collect::<Vec<T>>();
		self.replace_element(clip_v)
    }

}
