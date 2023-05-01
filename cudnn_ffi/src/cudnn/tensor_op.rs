use crate::*;
use crate::{cudnn::context::*, cudnn::tensor::*};
use std::os::raw::*;

impl Context {

    fn opTensor(&self, op:TensorOpDescriptor,
		a:&Tensor, b:&Tensor, scales:[f32;3], dst_shape:&[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
	let alpha1:*const c_float = &scales[0];
	let alpha2:*const c_float = &scales[1];
	let beta:*const c_float = &scales[2];
	let dstTensor = Tensor::new(a.getDataType(),dst_shape)?;

	let result = unsafe { cudnnOpTensor(self.handle,
					    op.opTensorDesc,
					    alpha1 as *const c_void,
					    a.descriptor,
					    a.gpu_memory,
					    alpha2 as *const c_void,
					    b.descriptor,
					    b.gpu_memory,
					    beta as *const c_void,
					    dstTensor.descriptor,
					    dstTensor.gpu_memory) };
	if result != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    println!("unable to operation tensor");
	    return Err(Box::<CUDNNStatus>::new(result.into()));
	}
	Ok(dstTensor)
    }

    pub fn add(&self, x:&Tensor, y:&Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
	assert_eq!(x.shape(), y.shape());
	let opTensorDesc = TensorOpDescriptor::new(crate::CUDNNTensorOp::ADD,
						   x.getDataType(),
						   true)?;
	self.opTensor(opTensorDesc, x, y, [1.0,1.0,1.0], x.shape())
    }

    pub fn matmul(&self, x:&Tensor, y:&Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
	let opTensorDesc = TensorOpDescriptor::new(crate::CUDNNTensorOp::MUL,
						   x.getDataType(),
						   true)?;
	let mut dst_shape = x.shape().to_vec();
	dst_shape[2] = x.shape()[2];
	dst_shape[3] = y.shape()[3];
	self.opTensor(opTensorDesc, x, y, [1.0,1.0,1.0], &dst_shape)
    }
}
