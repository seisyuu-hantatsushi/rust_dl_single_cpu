use crate::*;

use std::{fmt,ptr};
use std::os::raw::*;

pub struct TensorAttribute {
    shape: Vec<usize>,
    dataType: CUDNNDataType
}

pub struct Tensor {
    pub descriptor: cudnnTensorDescriptor_t,
    pub gpu_memory: *mut std::os::raw::c_void,
    attribute : TensorAttribute
}

impl Drop for Tensor {
    fn drop(&mut self) {
	unsafe {
	    cudaFree(self.gpu_memory);
	    cudnnDestroyTensorDescriptor(self.descriptor)
	};
    }
}

fn fmt_recursive<'a, T>(depth:usize, shape:&'a[usize], v:&'a[T]) -> String
where T:std::fmt::Display {
    let indent_element = "    ".to_string();
    let indent = if depth == 0 { "".to_string() } else { (0..depth).fold("".to_string(),|s, _| s + &indent_element) };
    if shape.len() > 1 {
	let stride = shape[1..shape.len()].iter().fold(1,|prod, x| prod * (*x));
		let mut l = indent.clone() + "[\n";
		for i in 0..shape[0] {
			l = l + &fmt_recursive(depth+1, &shape[1..shape.len()], &v[stride*i..(stride*(i+1))]) + "\n";
		}
		l+&indent+"],"
    }
    else {
		indent + "[" + &v.iter().fold("".to_string(),|s, x| s + &x.to_string()+",").clone() + "]"
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let shape = self.shape();
	let size = shape.iter().fold(1, |p,&e| { p*e });
	let host_buffer:Vec<f32> = self.getValues().unwrap();
	let mut disp = format!("Tensor [");
	for s in shape {
	    disp = format!("{}{},", disp, s);
	}
	disp = format!("{}]\n",disp);
	disp = format!("{}{}",disp,fmt_recursive(0, &shape[0..shape.len()], &host_buffer));
	write!(f, "{}", disp)
    }
}

impl Tensor {

    pub fn new(dataType:CUDNNDataType,
	       shape:&[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
	let mut descriptor = ptr::null_mut();
	let mut gpu_memory: *mut std::os::raw::c_void = ptr::null_mut();
	let result = unsafe { cudnnCreateTensorDescriptor(&mut descriptor) };
	if result != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    return Err(Box::<CUDNNStatus>::new(result.into()));
	}
	let sizeOfDataType = dataType.size();
	let sizeOfDataSize = shape.iter().fold(1, |p,s| { p * s }) * sizeOfDataType;
	let dims:Vec<std::os::raw::c_int> = shape.iter().map(|&d| { d as std::os::raw::c_int }).collect();
	fn calc_stride(dims:&[c_int]) -> Vec<c_int> {
	    let mut stride:Vec<c_int> = vec!();
	    if dims.len() == 1 {
		return vec![1];
	    }
	    else {
		let prod = dims[1..].iter().fold(1,|p,&s| { p*s });
		stride.push(prod);
		let sv = calc_stride(&dims[1..]);
		stride.extend(sv)
	    }
	    stride
	}
	let strides:Vec<c_int> = calc_stride(&dims);

	let len = if shape.len() < 4 { 4 } else { shape.len() };
	let dims:Vec<std::os::raw::c_int> = if dims.len() == 1 {
	    vec![1,1,1,1]
	}
	else if dims.len() == 2 {
	    vec![1,1,dims[1],dims[0]]
	}
	else if dims.len() == 3 {
	    vec![1,dims[0],dims[1],dims[2]]
	}
	else {
	    dims
	};
	let strides:Vec<std::os::raw::c_int> = if strides.len() == 1 {
	    vec![1,1,1,strides[0]]
	}
	else if strides.len() == 2 {
	    vec![1,1,strides[1],strides[0]]
	}
	else if strides.len() == 3 {
	    vec![1,strides[0],strides[1],strides[2]]
	}
	else {
	    strides
	};

	//println!("len: {:?}", len);
	//println!("dims: {:?}", dims);
	//println!("stride: {:?}", strides);
	let result = unsafe { cudnnSetTensorNdDescriptor(descriptor, dataType.into(), len as c_int, dims.as_ptr(), strides.as_ptr()) };
	if result != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    unsafe { cudnnDestroyTensorDescriptor(descriptor) };
	    return Err(Box::<CUDNNStatus>::new(result.into()));
	}

	let result = unsafe { cudaMalloc(&mut gpu_memory, sizeOfDataSize as size_t) };
	if result != cudaError_cudaSuccess {
	    unsafe { cudnnDestroyTensorDescriptor(descriptor) };
	    return Err(Box::<CUDAError>::new(result.into()));
	}
	let result = unsafe { cudaMemset(gpu_memory, 0, sizeOfDataSize as u64) };
	if result != cudaError_cudaSuccess {
	    unsafe {
		cudaFree(gpu_memory);
		cudnnDestroyTensorDescriptor(descriptor);
	    };
	    return Err(Box::<CUDAError>::new(result.into()));
	}
	Ok(Tensor {
	    descriptor,
	    gpu_memory,
	    attribute : TensorAttribute {
		shape: shape.to_vec(),
		dataType
	    }
	})
    }

    pub fn getDataType(&self) -> CUDNNDataType {
	self.attribute.dataType
    }

    pub fn shape(&self) -> &[usize] {
	&self.attribute.shape
    }

    pub fn setValues(&mut self, buffer:Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
	let result = unsafe {
	    cudaMemcpy(self.gpu_memory,
		       buffer.as_ptr() as *const c_void,
		       (buffer.len()*size_of::<f32>()) as size_t,
		       (CUDAMemcpyKind::HostToDevice).into())
	};
	if result != cudaError_cudaSuccess {
	    return Err(Box::<CUDAError>::new(result.into()));
	}
	Ok(())
    }

    pub fn getValues(&self) -> Result<Vec<f32>,  Box<dyn std::error::Error>> {
	let shape = &self.attribute.shape;
	let size = shape.iter().fold(1, |p,&e| { p*e });
	let mut host_buffer:Vec<f32> = vec![0f32;size];
	let result = unsafe {
	    cudaMemcpy(host_buffer.as_mut_ptr() as *mut c_void,
		       self.gpu_memory,
		       (host_buffer.len()*size_of::<f32>()) as size_t,
		       (CUDAMemcpyKind::DeviceToHost).into())
	};
	if result != cudaError_cudaSuccess {
	    return Err(Box::<CUDAError>::new(result.into()));
	}
	Ok(host_buffer)
    }

    pub fn from_vector(dataType:CUDNNDataType,
		       shape:&[usize],
		       datas:Vec<f32>) -> Result<Tensor, Box<dyn std::error::Error>> {
	let mut newTensor = Tensor::new(dataType, shape)?;
	newTensor.setValues(datas)?;
	Ok(newTensor)
    }
}

pub struct TensorOpDescriptor {
    pub opTensorDesc:cudnnOpTensorDescriptor_t
}

impl TensorOpDescriptor {
    pub fn new(op:CUDNNTensorOp,
	       dataType:CUDNNDataType,
	       nanPropagatioin:bool) -> Result<TensorOpDescriptor, Box<dyn std::error::Error>> {
	let mut opDesc = ptr::null_mut();
	let result = unsafe { cudnnCreateOpTensorDescriptor(&mut opDesc) };
	if result != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    return Err(Box::<CUDNNStatus>::new(result.into()));
	};
	let result = unsafe { cudnnSetOpTensorDescriptor(opDesc, op.into(),
							 dataType.into(),
							 if nanPropagatioin {
							     cudnnNanPropagation_t_CUDNN_PROPAGATE_NAN
							 }
							 else {
							     cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN
							 }) };
	if result != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    return Err(Box::<CUDNNStatus>::new(result.into()));
	};

	Ok(TensorOpDescriptor {
	    opTensorDesc: opDesc
	})
    }
}

impl Drop for TensorOpDescriptor {
    fn drop(&mut self) {
	unsafe {
	    cudnnDestroyOpTensorDescriptor(self.opTensorDesc);
	};
    }
}
