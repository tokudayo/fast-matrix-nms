#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>
#include <torch/extension.h>

enum Ops
{
    gaussian,
    linear
};

const char mult = 4;
const int dim = 1 << mult;
dim3 const blockSize(dim, dim);
const Ops op = Ops::gaussian;

template <typename T>
__device__ inline T boxIoU(
    T const *const a,
    T const *const b)
{
    T width = max(
        min(a[2], b[2]) - max(a[0], b[0]), (T)0);
    T height = max(
        min(a[3], b[3]) - max(a[1], b[1]), (T)0);
    T Sa = (a[2] - a[0]) * (a[3] - a[1]);
    T Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return width * height / (Sa + Sb - width * height);
}

template <typename T>
__device__ inline void warpReduce(volatile T *sdata, const int lid)
{
    if (dim == 32)
        sdata[lid] = max(sdata[lid], sdata[lid + 16]);
    sdata[lid] = max(sdata[lid], sdata[lid + 8]);
    sdata[lid] = max(sdata[lid], sdata[lid + 4]);
    sdata[lid] = max(sdata[lid], sdata[lid + 2]);
    sdata[lid] = max(sdata[lid], sdata[lid + 1]);
}

template <typename T>
__global__ void maxReduceLastWarp_(
    const int g_stride,
    T *gdata,
    T *out_tensor)
{
    // This should give a tiny performance boost, but I just leave it here for now
    const int tid = threadIdx.y * dim + threadIdx.x;

    __shared__ T mat[dim * dim];
    mat[tid] = gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + threadIdx.x];

    __syncwarp();
    if (threadIdx.x < dim >> 1)
        warpReduce<T>(mat + threadIdx.y * dim, threadIdx.x);

    if (threadIdx.x == 0)
    {
        out_tensor[blockIdx.y * dim + threadIdx.y] = mat[threadIdx.y * dim];
    }
}

template <typename T>
__global__ void iouMatrixFirstMaxReduce_(
    const int n_boxes,
    const int g_stride,
    const T *dev_boxes,
    T *dev_iou,
    T *gdata)
{
    const int tid = threadIdx.y * dim + threadIdx.x;

    if (blockIdx.y < blockIdx.x)
        return;

    __shared__ T boxes[dim * 2 * 4];
    __shared__ T iou[dim * dim];

    if (tid < dim * 4)
    {
        boxes[tid] =
            dev_boxes[blockIdx.y * dim * 4 + tid];
    }
    else if (tid < dim * 8)
    {
        boxes[tid] =
            dev_boxes[blockIdx.x * dim * 4 + tid - dim * 4];
    }
    else
    {
        int s = tid - dim * 8;
        while (s < dim * dim)
        {
            iou[s] = 0;
            s += dim * (dim - 8);
        }
    }
    __syncthreads();

    // Calculate IoU matrix
    // if (blockIdx.x == blockIdx.y && threadIdx.x >= threadIdx.y)
    //     return;
    if (blockIdx.x != blockIdx.y || threadIdx.x < threadIdx.y)
        iou[tid] = boxIoU<T>(boxes + threadIdx.y * 4, boxes + (dim + threadIdx.x) * 4);
    dev_iou[(blockIdx.y * dim + threadIdx.y) * n_boxes + blockIdx.x * dim + threadIdx.x] = iou[tid];

    // First max reduction
    if (threadIdx.x < dim >> 1)
        warpReduce<T>(iou + threadIdx.y * dim, threadIdx.x);
    if (threadIdx.x == 0)
    {
        gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x] = iou[threadIdx.y * dim];
    }
}

template <typename T>
__global__ void maxReduceFirstIter_(
    const int g_stride,
    const int remainder,
    T *gdata)
{
    // Scale factor at first step is dim
    if (blockIdx.x * dim > blockIdx.y)
        return;

    const int tid = threadIdx.y * dim + threadIdx.x;
    __shared__ T mat[dim * dim];
    if (blockIdx.x != gridDim.x - 1)
        mat[tid] = gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x * dim + threadIdx.x];
    else
    {
        if (threadIdx.x < remainder)
            mat[tid] = gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x * dim + threadIdx.x];
        else
            mat[tid] = 0;
    }
    __syncwarp();

    if (threadIdx.x < dim >> 1)
        warpReduce<T>(mat + threadIdx.y * dim, threadIdx.x);
    if (threadIdx.x == 0)
    {
        gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x] = mat[threadIdx.y * dim];
    }
}

template <typename T>
__global__ void maxReduce_(
    const int g_stride,
    const int col_scale_factor,
    T *gdata)
{
    if (blockIdx.x * col_scale_factor > blockIdx.y)
        return;

    const int tid = threadIdx.y * dim + threadIdx.x;
    __shared__ T mat[dim * dim];
    mat[tid] = gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x * dim + threadIdx.x];
    __syncwarp();

    if (threadIdx.x < dim >> 1)
        warpReduce<T>(mat + threadIdx.y * dim, threadIdx.x);
    if (threadIdx.x == 0)
    {
        gdata[(blockIdx.y * dim + threadIdx.y) * g_stride + blockIdx.x] = mat[threadIdx.y * dim];
    }
}

template <typename T>
__global__ void broadcastOpsAndSelect_(
    const int n_boxes,
    T *dev_iou,
    const T *iou_max,
    const T *scores,
    bool *drop,
    const float threshold)
{
    if (blockIdx.y < blockIdx.x)
        return;

    __shared__ T scalars[dim * 2];

    if (threadIdx.y == 0)
        scalars[threadIdx.x] = iou_max[blockIdx.x * blockDim.x + threadIdx.x];
    if (threadIdx.y == 1)
        scalars[dim + threadIdx.x] = scores[blockIdx.y * blockDim.y + threadIdx.x];
    __syncthreads();

    // if (blockIdx.x == blockIdx.y && threadIdx.x >= threadIdx.y)
    // return;

    switch (op)
    {
    case Ops::gaussian:
        if (
            exp((scalars[threadIdx.x] * scalars[threadIdx.x] -
                 dev_iou[(blockIdx.y * dim + threadIdx.y) * n_boxes + blockIdx.x * dim + threadIdx.x] * dev_iou[(blockIdx.y * dim + threadIdx.y) * n_boxes + blockIdx.x * dim + threadIdx.x]) *
                2) *
                scalars[dim + threadIdx.y] <
            threshold)
            drop[blockIdx.y * blockDim.y + threadIdx.y] = true;
        break;

    case Ops::linear:
        if (
            (1 - dev_iou[(blockIdx.y * dim + threadIdx.y) * n_boxes + blockIdx.x * dim + threadIdx.x]) / (1 - scalars[threadIdx.x]) * scalars[dim + threadIdx.y] < threshold)
            drop[blockIdx.y * blockDim.y + threadIdx.y] = true;
        break;
    }
}

at::Tensor fast_matrix_nms_cuda(
    const at::Tensor &dets,
    const at::Tensor &scores,
    const float score_threshold)
{
    TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

    TORCH_CHECK(
        dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
    TORCH_CHECK(
        dets.size(1) == 4,
        "boxes should have 4 elements in dimension 1, got ",
        dets.size(1));
    TORCH_CHECK(
        scores.dim() == 1,
        "scores should be a 1d tensor, got ",
        scores.dim(),
        "D");
    TORCH_CHECK(
        dets.size(0) == scores.size(0),
        "boxes and scores should have same number of elements in ",
        "dimension 0, got ",
        dets.size(0),
        " and ",
        scores.size(0))

    if (dets.numel() == 0)
    {
        return at::empty({0}, dets.options().dtype(at::kFloat));
    }

    // Sort the scores in descending order
    at::Tensor scores_sorted, order_t;
    std::tie(scores_sorted, order_t) = scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true);
    auto dets_sorted = dets.index_select(0, order_t).contiguous();

    const int N = dets.size(0);
    const int bDim = N >> mult + (N % dim != 0);
    dim3 gridSize(bDim, bDim);

    at::Tensor iou_matrix = at::empty({N, N}, dets.options().dtype(at::kFloat));
    int ret_dim = N >> mult;
    ret_dim = ((ret_dim + dim - 1) >> mult) * dim;
    const int g_stride = ret_dim;
    at::Tensor g_out = at::zeros({N, ret_dim}, dets.options().dtype(at::kFloat));

    AT_DISPATCH_FLOATING_TYPES(
        dets_sorted.scalar_type(), "iou_first_max", [&]
        { iouMatrixFirstMaxReduce_<scalar_t><<<gridSize, blockSize>>>(
              N,
              g_stride,
              dets_sorted.data_ptr<scalar_t>(),
              iou_matrix.data_ptr<scalar_t>(),
              g_out.data_ptr<scalar_t>()); });
    // return g_out;
    int scale_factor = 1;
    while (ret_dim > 1)
    {
        ret_dim = (ret_dim + dim - 1) >> mult;
        dim3 ret_grid_dim(ret_dim, N >> mult);
        scale_factor <<= mult;

        if (scale_factor == dim)
        {
            int remainder = (N >> mult) % dim;
            AT_DISPATCH_FLOATING_TYPES(
                dets_sorted.scalar_type(), "max_reduction_first", [&]
                { maxReduceFirstIter_<scalar_t><<<ret_grid_dim, blockSize>>>(
                      g_stride,
                      remainder,
                      g_out.data_ptr<scalar_t>()); });
        }
        else
            AT_DISPATCH_FLOATING_TYPES(
                dets_sorted.scalar_type(), "max_reduction", [&]
                { maxReduce_<scalar_t><<<ret_grid_dim, blockSize>>>(
                      g_stride,
                      scale_factor,
                      g_out.data_ptr<scalar_t>()); });
    }
    // Get max IOU
    // at::Tensor ious_rmax = at::zeros({N}, dets.options().dtype(at::kFloat));
    auto ious_rmax = g_out.slice(/*dim=*/1, /*start=*/0, /*end=*/1).squeeze(/*dim=*/1).contiguous();
    // return ious_rmax;
    at::Tensor drop = at::zeros({N}, dets.options().dtype(at::kBool));

    AT_DISPATCH_FLOATING_TYPES(
        dets_sorted.scalar_type(), "score_select", [&]
        { broadcastOpsAndSelect_<scalar_t><<<gridSize, blockSize>>>(
              N,
              iou_matrix.data_ptr<scalar_t>(),
              ious_rmax.data_ptr<scalar_t>(),
              scores_sorted.data_ptr<scalar_t>(),
              drop.data_ptr<bool>(),
              score_threshold); });

    auto order_keep = order_t.masked_select(at::logical_not(drop));
    return order_keep;
}
