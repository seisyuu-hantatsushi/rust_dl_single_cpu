use crate::*;
use std::rc::Rc;

pub mod flags {
    pub const SchedAuto:u32  = cuda_sys::CUctx_flags_enum_CU_CTX_SCHED_AUTO;
    pub const SchedSpin:u32  = cuda_sys::CUctx_flags_enum_CU_CTX_SCHED_SPIN;
    pub const SchedYeild:u32 = cuda_sys::CUctx_flags_enum_CU_CTX_SCHED_YIELD;
    pub const SchedBlockingSync:u32 = cuda_sys::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC;
    pub const SchedMask:u32         = cuda_sys::CUctx_flags_enum_CU_CTX_SCHED_MASK;
    pub const MapHost:u32           = cuda_sys::CUctx_flags_enum_CU_CTX_MAP_HOST;
    pub const CoreDumpEnable:u32    = cuda_sys::CUctx_flags_enum_CU_CTX_COREDUMP_ENABLE;
    pub const UserCoreDumpEnable:u32 = cuda_sys::CUctx_flags_enum_CU_CTX_USER_COREDUMP_ENABLE;
}

#[derive(Debug)]
pub struct Context {
    device:  i32,
    context: CUcontext
}

pub type SharableContext = Rc<Context>;

impl Context {

    pub fn new(device: cuda::device::Device,
	       flags: u32) -> Result<SharableContext,CUDAError>{
	let mut context: CUcontext = std::ptr::null_mut();
	let result:CUDAError = unsafe {
	    //created context pushed to current when this function is called.
	    cuCtxCreate_v2(&mut context, flags, device.as_raw())
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
/*
	unsafe {
	    let mut poped_context: CUcontext = std::ptr::null_mut();
	    cuCtxPopCurrent_v2(&mut poped_context);
	}
*/
	Ok(Rc::new(Context {
	    device: device.as_raw(),
	    context
	}))
    }

    pub fn get_current() -> Result<Option<SharableContext>,CUDAError> {
	let mut context:CUcontext = std::ptr::null_mut();
	let mut device:i32 = 0;

	let result:CUDAError = unsafe {
	    cuCtxGetDevice(&mut device)
	}.into();

	if result != CUDAError::Success {
	    return Err(result);
	}

	let result:CUDAError = unsafe {
	    cuCtxGetCurrent(&mut context)
	}.into();

	if result != CUDAError::Success {
	    return Err(result);
	}

	if context == std::ptr::null_mut() {
	    Ok(None)
	}
	else {
	    Ok(Some(Rc::new(Context {
		device,
		context
	    })))
	}
    }

    pub fn push(&self) -> Result<(), CUDAError> {
	let result:CUDAError = unsafe {
	    cuCtxPushCurrent_v2(self.context)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(())
    }

    pub fn pop() -> Result<Option<SharableContext>,CUDAError> {
	let mut poped_context: CUcontext = std::ptr::null_mut();
	let mut device:i32 = 0;

	let result:CUDAError = unsafe {
	    cuCtxGetDevice(&mut device)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}

	let result: CUDAError = unsafe {
	    cuCtxPopCurrent_v2(&mut poped_context)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}

	if poped_context == std::ptr::null_mut() {
	    Ok(None)
	}
	else {
	    Ok(Some(Rc::new(Context {
		device,
		context: poped_context
	    })))
	}
    }
}

impl Drop for Context {
    fn drop(&mut self) {
	unsafe { cuCtxDestroy_v2(self.context); }
    }
}
