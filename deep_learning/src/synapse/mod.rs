pub mod synapse;
pub mod synapse_reshapes;
pub mod synapse_hadamard_product;
pub mod synapse_matrix_product;
pub mod synapse_mul_rank0;
pub mod synapse_div_rank0;
pub mod synapse_pow;
pub mod synapse_log;
pub mod synapse_trigonometric;
pub mod synapse_error_funcs;

pub use synapse::{Synapse, SynapseOption, SynapseNode, NNSynapseNode};
