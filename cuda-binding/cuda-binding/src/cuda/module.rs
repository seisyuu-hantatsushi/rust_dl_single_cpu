
use crate::*;
use std::os::raw::*;
use std::ffi::CString;
use std::env;

#[derive(Debug)]
pub struct Module {
    module: CUmodule
}

#[derive(Debug)]
pub struct Function {
    function: CUfunction
}

impl Module {

    pub fn new(fatbin:&[u8]) -> Result<Module,CUDAError> {
	let mut module:CUmodule = std::ptr::null_mut();
	let result:CUDAError = unsafe {
	    cuModuleLoadFatBinary(&mut module, fatbin.as_ptr() as *const c_void)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(Module {
	    module
	})
    }

    pub fn get_function(&self, name:&str) -> Result<Function, CUDAError> {
	let mut function:CUfunction = std::ptr::null_mut();
	let cname = CString::new(name).unwrap();
	let result:CUDAError = unsafe {
	    cuModuleGetFunction(&mut function, self.module, cname.as_ptr())
	}.into();
	//println!("function handle: {:?}", function);
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(Function {
	    function
	})
    }
}

impl Drop for Module {
    fn drop(&mut self) {
	unsafe {
	    cuModuleUnload(self.module);
	};
    }
}

impl Function {
    pub fn as_raw(&self) -> CUfunction {
	self.function
    }
}
