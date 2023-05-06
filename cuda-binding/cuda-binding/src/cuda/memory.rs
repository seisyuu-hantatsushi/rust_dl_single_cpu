use crate::*;
use std::mem;
use std::marker::PhantomData;
use std::os::raw::*;

#[derive(Debug)]
pub struct Memory<T> {
    phantom: PhantomData<T>,
    device_ptr: *mut c_void,
    element_size: usize
}

impl<T> Memory<T> {

    pub fn malloc(element_size:usize) -> Result<Self, CUDAError> {
	let mut ptr:*mut c_void = std::ptr::null_mut();
	let request_size = element_size*mem::size_of::<T>();
	let result:CUDAError = unsafe {
	    cudaMalloc(&mut ptr, request_size as size_t)
	}.into();
	//println!("malloc ptr = {:?}", ptr);
	if result != CUDAError::Success {
	    return Err(result);
	}

	Ok(Memory {
	    phantom: PhantomData,
	    device_ptr: ptr,
	    element_size,
	})
    }

    pub fn raw_ptr(&self) -> *mut c_void {
	self.device_ptr
    }

    pub fn from_host(&mut self, data:&Vec<T>) -> Result<(), CUDAError> {
	let area_size = data.len()*mem::size_of::<T>();
	let result:CUDAError = unsafe {
	    cudaMemcpy(self.device_ptr,
		       data.as_ptr() as *const c_void,
		       area_size as u64,
		       cudaMemcpyKind_cudaMemcpyHostToDevice)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(())
    }

    pub fn to_host(&mut self) -> Result<Vec<T>, CUDAError> {
	let mut host_mem = Vec::with_capacity(self.element_size);
	let result:CUDAError = unsafe {
	    host_mem.set_len(self.element_size);
	    cudaMemcpy(host_mem.as_ptr() as *mut c_void,
		       self.device_ptr,
		       (self.element_size*mem::size_of::<T>()).try_into().unwrap(),
		       cudaMemcpyKind_cudaMemcpyDeviceToHost)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(host_mem)
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
	unsafe { cudaFree(self.device_ptr) };
    }
}
