#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::os::raw::*;

    #[test]
    fn add_tensor() {
	let mut handle: cudnnHandle_t = ptr::null_mut();
	unsafe {
	    match cudnnCreate(&mut handle) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN context.")
	    }

	    let mut xDesc:cudnnTensorDescriptor_t = ptr::null_mut();
	    let mut yDesc:cudnnTensorDescriptor_t = ptr::null_mut();
	    let mut zDesc:cudnnTensorDescriptor_t = ptr::null_mut();

	    match cudnnCreateTensorDescriptor(&mut xDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }
	    match cudnnCreateTensorDescriptor(&mut yDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }
	    match cudnnCreateTensorDescriptor(&mut zDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }

	    let batchSize:c_int = 1;
	    let channels:c_int  = 3;
	    let height:c_int = 4;
	    let width:c_int = 4;
	    let dims:[c_int;4] = [batchSize, channels, height, width];
	    let strides:[c_int;4] = [channels*height*width,height*width, width, 1];
	    cudnnSetTensorNdDescriptor(xDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    cudnnSetTensorNdDescriptor(yDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    cudnnSetTensorNdDescriptor(zDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    let dataSize:usize = (batchSize*channels*height*width) as usize;
	    let xDataHost:Vec<c_float> = (0..dataSize).map(|i| i as f32).collect();
	    let yDataHost:Vec<c_float> = (0..dataSize).map(|i| (i*2) as f32).collect();
	    let mut zDataHost:Vec<c_float> = vec![0.0;dataSize];
	    let mut xDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();
	    let mut yDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();
	    let mut zDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();

	    let alloc_size:size_t = (dataSize*std::mem::size_of::<c_float>()).try_into().unwrap();
	    cudaMalloc(&mut xDataDevice,alloc_size);
	    cudaMalloc(&mut yDataDevice,alloc_size);
	    cudaMalloc(&mut zDataDevice,alloc_size);

	    cudaMemcpy(xDataDevice,
		       xDataHost.as_ptr() as *const c_void,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyHostToDevice);
	    cudaMemcpy(yDataDevice,
		       yDataHost.as_ptr() as *const c_void,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyHostToDevice);
	    let mut opTensorDesc:cudnnOpTensorDescriptor_t = ptr::null_mut();
	    cudnnCreateOpTensorDescriptor(&mut opTensorDesc);
	    cudnnSetOpTensorDescriptor(opTensorDesc,
				       cudnnOpTensorOp_t_CUDNN_OP_TENSOR_ADD,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       cudnnNanPropagation_t_CUDNN_PROPAGATE_NAN);
	    let alpha1:[c_float;1] = [1.0];
	    let alpha2:[c_float;1] = [1.0];
	    let beta:[c_float;1]   = [1.0];

	    cudnnOpTensor(handle, opTensorDesc,
			  alpha1.as_ptr() as *const c_void,
			  xDesc,
			  xDataDevice,
			  alpha2.as_ptr() as *const c_void,
			  yDesc,
			  yDataDevice,
			  beta.as_ptr() as  *const c_void,
			  zDesc,
			  zDataDevice);

	    cudaMemcpy(zDataHost.as_ptr() as *mut c_void,
		       zDataDevice,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyDeviceToHost);

	    for (i,v) in zDataHost.iter().enumerate() {
		println!("zDataHost[{i}]={v}");
		assert_eq!(*v, xDataHost[i]+yDataHost[i]);
	    }

	    cudaFree(xDataDevice);
	    cudaFree(yDataDevice);
	    cudaFree(zDataDevice);

	    cudnnDestroyTensorDescriptor(xDesc);
	    cudnnDestroyTensorDescriptor(yDesc);
	    cudnnDestroyTensorDescriptor(zDesc);

	    match cudnnDestroy(handle) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to destroy cuDNN context.")
	    }
	}
    }
}
