#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

at::Tensor fast_matrix_nms_cuda(
    const at::Tensor &dets,
    const at::Tensor &scores,
    const float score_threshold);

auto fast_matrix_nms(
    torch::Tensor &boxes,
    torch::Tensor &scores,
    float score_threshold)
{
  CHECK_INPUT(boxes);
  CHECK_INPUT(scores);

  torch::NoGradGuard no_grad;
  return fast_matrix_nms_cuda(boxes, scores, score_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fast_matrix_nms", &fast_matrix_nms, "Fast Matrix NMS (CUDA)");
}