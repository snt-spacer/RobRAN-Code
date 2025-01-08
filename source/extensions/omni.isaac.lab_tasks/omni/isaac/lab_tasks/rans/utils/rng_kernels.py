# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import warp as wp


@wp.func
def set_state(seed: wp.int32) -> wp.uint32:
    """Set the seed for the random number generator.
    Args:
        seed: The seed for the random number generator.
    """
    return wp.rand_init(seed)


@wp.kernel
def set_states(
    seed: wp.array(dtype=wp.int32),
    old_seeds: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
):
    """Set the seeds for the random number generator.
    Args:
        seed: The new seeds to be assigned to the selected environments.
        old_seeds: The old seed for each environment.
        states: The state for each environment.
        ids: The ids of the selected environments."""
    tid = wp.tid()
    states[ids[tid]] = set_state(seed[tid])
    old_seeds[ids[tid]] = seed[tid]


###################
# UNIFORM
###################


@wp.kernel
def rand_uniform_1D(
    low: wp.float32,
    high: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample from a uniform distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.randf(states[ids[tid]], low, high)
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_uniform_2D(
    low: wp.float32,
    high: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a uniform distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.randf(states[ids[i]] + wp.uint32(j), low, high)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_uniform_3D(
    low: wp.float32,
    high: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=float, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a uniform distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.randf(states[ids[i]] + wp.uint32(j * shape[1] + k), low, high)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def uniform_single(
    low: float, high: float, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """Sample from a uniform distribution.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values.
    """
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.float32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_uniform_1D,
                dim=kernel_shape[0],
                inputs=[low, high, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_uniform_2D,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_uniform_3D,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


###################
# UNIFORM TENSOR
###################


@wp.kernel
def rand_uniform_1D_tensorized(
    low: wp.array(dtype=wp.float32),
    high: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample from a uniform distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution for each environments.
        high: The upper bound of the uniform distribution for each environments.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.randf(states[ids[tid]], low[tid], high[tid])
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_uniform_2D_tensorized(
    low: wp.array(dtype=wp.float32),
    high: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a uniform distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution for each environments.
        high: The upper bound of the uniform distribution for each environments.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.randf(states[ids[i]] + wp.uint32(j), low[i], high[i])
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_uniform_3D_tensorized(
    low: wp.array(dtype=wp.float32),
    high: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=float, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a uniform distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution for each environments.
        high: The upper bound of the uniform distribution for each environments.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.randf(states[ids[i]] + wp.uint32(j * shape[1] + k), low[i], high[i])
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def uniform_tensorized(
    low: wp.array,
    high: wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
) -> wp.array:
    """Sample from a uniform distribution.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values.
    """
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.float32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_uniform_1D_tensorized,
                dim=kernel_shape[0],
                inputs=[low, high, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_uniform_2D_tensorized,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_uniform_3D_tensorized,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def uniform(
    low: float | wp.array,
    high: float | wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
):
    if isinstance(low, wp.array) or isinstance(high, wp.array):
        if isinstance(low, float) or isinstance(high, float):
            raise ValueError("The high value must be a tensor if the low value is a tensor.")
        output = uniform_tensorized(low, high, states, new_states, ids, shape, device=device)
    elif isinstance(low, float) or isinstance(high, float):
        if isinstance(low, wp.array) or isinstance(high, wp.array):
            raise ValueError("The low value must be a tensor if the high value is a tensor.")
        output = uniform_single(low, high, states, new_states, ids, shape, device=device)
    return output


###################
# SIGN
###################


@wp.func
def rand_sign(state: wp.uint32) -> wp.int32:
    """Sample a random sign.
    Args:
        state: The state for the random number generator.
    Returns:
        The sampled sign. (-1 or 1)"""
    rand_num = wp.randf(state)
    if rand_num < 0.5:
        return wp.int32(-1)
    else:
        return wp.int32(1)


@wp.kernel
def rand_sign_1D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Sample a random sign as an integer. 1D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = rand_sign(states[ids[tid]])
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_sign_2D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.uint32,
):
    """Sample a random sign as an integer. 2D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = rand_sign(states[ids[i]] + wp.uint32(j))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_sign_3D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample a random sign as an integer. 3D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = rand_sign(states[ids[i]] + wp.uint32(j * shape[1] + k))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_sign_1Df(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample a random sign as a float. 1D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.float32(rand_sign(states[ids[tid]]))
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_sign_2Df(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample a random sign as a float. 2D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.float32(rand_sign(states[ids[i]] + wp.uint32(j)))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_sign_3Df(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample a random sign as a float. 3D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.float32(rand_sign(states[ids[i]] + wp.uint32(j * shape[1] + k)))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def rand_sign_int(states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda") -> wp.array:
    """
    Sample a random sign as a integer.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.int32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_sign_1D,
                dim=kernel_shape[0],
                inputs=[states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_sign_2D,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_sign_3D,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def rand_sign_float(
    states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """
    Sample a random sign as a float.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.float32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_sign_1Df,
                dim=kernel_shape[0],
                inputs=[states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_sign_2Df,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_sign_3Df,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def rand_sign_fn(
    states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], dtype: str, device="cuda"
) -> wp.array:
    """Random sign function. Uses the correct kernel based on the desired dtype.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        dtype: The data type of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    if dtype == "int":
        return rand_sign_int(states, new_states, ids, shape, device)
    elif dtype == "float":
        return rand_sign_float(states, new_states, ids, shape, device)
    else:
        raise ValueError("Invalid dtype, must be 'int' or 'float'")


###################
# POISSON
###################
# Note we need to cast from uint32 to int32 for the output of the poisson function so that torch can handle it


@wp.kernel
def rand_poisson_1D(
    lam: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Sample from a poisson distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.int32(wp.poisson(states[ids[tid]], lam))
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_poisson_2D(
    lam: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a poisson distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.int32(wp.poisson(states[ids[i]] + wp.uint32(j), lam))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_poisson_3D(
    lam: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a poisson distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.int32(wp.poisson(states[ids[i]] + wp.uint32(j * shape[1] + k), lam))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_poisson_1D_tensorized(
    lam: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Sample from a poisson distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.int32(wp.poisson(states[ids[tid]], lam[tid]))
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_poisson_2D_tensorized(
    lam: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a poisson distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.int32(wp.poisson(states[ids[i]] + wp.uint32(j), lam[i]))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_poisson_3D_tensorized(
    lam: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a poisson distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.int32(wp.poisson(states[ids[i]] + wp.uint32(j * shape[1] + k), lam[i]))
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def poisson_single(
    lam: float, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """
    Sample a random sign as a integer.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.int32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_poisson_1D,
                dim=kernel_shape[0],
                inputs=[lam, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_poisson_2D,
                dim=kernel_shape,
                inputs=[lam, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_poisson_3D,
                dim=kernel_shape,
                inputs=[lam, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def poisson_tensorized(
    lam: wp.array, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """
    Sample a random sign as a integer.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.int32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_poisson_1D_tensorized,
                dim=kernel_shape[0],
                inputs=[lam, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_poisson_2D_tensorized,
                dim=kernel_shape,
                inputs=[lam, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_poisson_3D_tensorized,
                dim=kernel_shape,
                inputs=[lam, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def poisson(
    lam: float | wp.array, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
):
    if isinstance(lam, wp.array):
        output = poisson_tensorized(lam, states, new_states, ids, shape, device=device)
    else:
        output = poisson_single(lam, states, new_states, ids, shape, device=device)
    return output


###################
# INT
###################


@wp.kernel
def rand_int_1D(
    low: wp.int32,
    high: wp.int32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Sample integer values from a uniform distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.randi(states[ids[tid]], low, high)
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_int_2D(
    low: wp.int32,
    high: wp.int32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.uint32,
):
    """Sample integer values from a uniform distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.randi(states[ids[i]] + wp.uint32(j), low, high)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_int_3D(
    low: wp.int32,
    high: wp.int32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample integer values from a uniform distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.randi(states[ids[i]] + wp.uint32(j * shape[1] + k), low, high)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_int_1D_tensorized(
    low: wp.array(dtype=wp.int32),
    high: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Sample integer values from a uniform distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = wp.randi(states[ids[tid]], low[tid], high[tid])
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_int_2D_tensorized(
    low: wp.array(dtype=wp.int32),
    high: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.uint32,
):
    """Sample integer values from a uniform distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = wp.randi(states[ids[i]] + wp.uint32(j), low[i], high[i])
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_int_3D_tensorized(
    low: wp.array(dtype=wp.int32),
    high: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample integer values from a uniform distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = wp.randi(states[ids[i]] + wp.uint32(j * shape[1] + k), low[i], high[i])
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def integer_single(
    low: int, high: int, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """
    Sample a random integer between two bounds.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.int32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_int_1D,
                dim=kernel_shape[0],
                inputs=[low, high, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_int_2D,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_int_3D,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def integer_tensorized(
    low: wp.array,
    high: wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
) -> wp.array:
    """
    Sample a random integer between two bounds.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        lam: The lambda parameter of the poisson distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.int32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_int_1D_tensorized,
                dim=kernel_shape[0],
                inputs=[low, high, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_int_2D_tensorized,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_int_3D_tensorized,
                dim=kernel_shape,
                inputs=[low, high, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def integer(
    low: int | wp.array,
    high: int | wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
):
    """Sample a random integer between two bounds.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    if isinstance(low, wp.array) or isinstance(high, wp.array):
        if isinstance(low, int) or isinstance(high, int):
            raise ValueError("The high value must be a tensor if the low value is a tensor.")
        output = integer_tensorized(low, high, states, new_states, ids, shape, device=device)
    elif isinstance(low, int) or isinstance(high, int):
        if isinstance(low, wp.array) or isinstance(high, wp.array):
            raise ValueError("The low value must be a tensor if the high value is a tensor.")
        output = integer_single(low, high, states, new_states, ids, shape, device=device)
    return output


###################
# NORMAL
###################


@wp.kernel
def rand_normal_1D(
    mean: wp.float32,
    std: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample from a normal distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = mean + wp.randn(states[ids[tid]]) * std
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_normal_2D(
    mean: wp.float32,
    std: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a normal distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = mean + wp.randn(states[ids[i]] + wp.uint32(j)) * std
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_normal_3D(
    mean: wp.float32,
    std: wp.float32,
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a normal distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = mean + wp.randn(states[ids[i]] + wp.uint32(j * shape[1] + k)) * std
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_normal_1D_tensorized(
    mean: wp.array(dtype=wp.float32),
    std: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample from a normal distribution. 1D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    output[tid] = mean[tid] + wp.randn(states[ids[tid]]) * std[tid]
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_normal_2D_tensorized(
    mean: wp.array(dtype=wp.float32),
    std: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a normal distribution. 2D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    output[i][j] = mean[i] + wp.randn(states[ids[i]] + wp.uint32(j)) * std[i]
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_normal_3D_tensorized(
    mean: wp.array(dtype=wp.float32),
    std: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a normal distribution. 3D version.
    The state for each environment is updated automatically.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    output[i][j][k] = mean[i] + wp.randn(states[ids[i]] + wp.uint32(j * shape[1] + k)) * std[i]
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def normal_single(
    mean: float, std: float, states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda"
) -> wp.array:
    """
    Samples a normal distribution between two bounds.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.float32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_normal_1D,
                dim=kernel_shape[0],
                inputs=[mean, std, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_normal_2D,
                dim=kernel_shape,
                inputs=[mean, std, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_normal_3D,
                dim=kernel_shape,
                inputs=[mean, std, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def normal_tensorized(
    mean: wp.array,
    std: wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
) -> wp.array:
    """
    Samples a normal distribution between two bounds.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape.
    The kernel will automatically update the state for each environment after sampling.
    Args:
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.float32, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_normal_1D_tensorized,
                dim=kernel_shape[0],
                inputs=[mean, std, states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_normal_2D_tensorized,
                dim=kernel_shape,
                inputs=[mean, std, states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_normal_3D_tensorized,
                dim=kernel_shape,
                inputs=[mean, std, states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs


def normal(
    mean: float | wp.array,
    std: float | wp.array,
    states: wp.array,
    new_states: wp.array,
    ids: wp.array,
    shape: tuple[int],
    device="cuda",
):
    if isinstance(mean, wp.array) or isinstance(std, wp.array):
        if isinstance(mean, float) or isinstance(std, float):
            raise ValueError("The high value must be a tensor if the low value is a tensor.")
        output = normal_tensorized(mean, std, states, new_states, ids, shape, device=device)
    elif isinstance(mean, float) or isinstance(std, float):
        if isinstance(mean, wp.array) or isinstance(std, wp.array):
            raise ValueError("The low value must be a tensor if the high value is a tensor.")
        output = normal_single(mean, std, states, new_states, ids, shape, device=device)
    return output


###################
# QUATERNION
###################


@wp.kernel
def rand_quaternion_1D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.quatf),
):
    """Sample from a unit sphere. 1D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor."""
    tid = wp.tid()
    vec3f_unit_sphere = wp.sample_unit_sphere(states[ids[tid]])
    angle = wp.randf(states[ids[tid]], 0.0, 2.0 * 4.0 * wp.atan(1.0))
    output[tid] = wp.quat_from_axis_angle(vec3f_unit_sphere, angle)
    new_states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_quaternion_2D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.quatf, ndim=2),
    offset: wp.uint32,
):
    """Sample from a unit sphere. 2D version.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 2D tensor. Used to calculate the correct state for each environment."""
    i, j = wp.tid()
    vec3f_unit_sphere = wp.sample_unit_sphere(states[ids[i]] + wp.uint32(j))
    angle = wp.randf(states[ids[i]] + wp.uint32(j), 0.0, 2.0 * 4.0 * wp.atan(1.0))
    output[i][j] = wp.quat_from_axis_angle(vec3f_unit_sphere, angle)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


@wp.kernel
def rand_quaternion_3D(
    states: wp.array(dtype=wp.uint32),
    new_states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.quatf, ndim=3),
    offset: wp.uint32,
    shape: wp.vec3i,
):
    """Sample from a unit sphere.
    The state for each environment is updated automatically.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        output: The output tensor.
        offset: The offset for the 3D tensor. Used to calculate the correct state for each environment.
        shape: The shape of the 3D tensor. Used to calculate the correct state for each environment."""
    i, j, k = wp.tid()
    vec3f_unit_sphere = wp.sample_unit_sphere(states[ids[i]] + wp.uint32(j * shape[1] + k))
    angle = wp.randf(states[ids[i]] + wp.uint32(j * shape[1] + k), 0.0, 2.0 * 4.0 * wp.atan(1.0))
    output[i][j][k] = wp.quat_from_axis_angle(vec3f_unit_sphere, angle)
    new_states[ids[i]] = states[ids[i]] + wp.uint32(offset)


def quaternion(states: wp.array, new_states: wp.array, ids: wp.array, shape: tuple[int], device="cuda") -> wp.array:
    """Sample from a unit sphere.
    Automatically uses the correct kernel based on the desired shape. It is important to note that the
    final shape is defined as: (ids.shape[0],) + shape + (4,).
    The kernel will automatically update the state for each environment after sampling.
    Args:
        states: The state for each environment.
        new_states: The new state for each environment.
        ids: The ids of the selected environments.
        shape: The shape of the output tensor.
        device: The device to be used for the computation.
    Returns:
        The sampled values."""
    offset = math.prod(shape)
    if shape[0] == 1:
        kernel_shape = (ids.shape[0],)
    else:
        kernel_shape = (ids.shape[0],) + shape
    outputs = wp.empty(kernel_shape, dtype=wp.quatf, device=device)
    match len(kernel_shape):
        case 1:
            wp.launch(
                kernel=rand_quaternion_1D,
                dim=kernel_shape[0],
                inputs=[states, new_states, ids, outputs],
                device=device,
            )
        case 2:
            wp.launch(
                kernel=rand_quaternion_2D,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset],
                device=device,
            )
        case 3:
            wp.launch(
                kernel=rand_quaternion_3D,
                dim=kernel_shape,
                inputs=[states, new_states, ids, outputs, offset, kernel_shape],
                device=device,
            )
        case _:
            raise ValueError("Invalid shape, must be 1, 2 or 3 dimensions")
    wp.copy(states, new_states)
    return outputs
