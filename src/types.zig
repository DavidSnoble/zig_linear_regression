pub const DataSet = struct { x: []f64, y: []f64 };
pub const Derivative = struct { dm: f64, db: f64 };
pub const Regression = struct { m: f64, b: f64 };
pub const HyperParameters = struct {
    epochs: usize,
    learning_rate: f64,
    name: []const u8,
    use_adaptive_lr: bool = false,
    decay_rate: f64 = 0.95,
    decay_steps: usize = 1000,
};
pub const EvaluationResult = struct {
    predictions: std.array_list.Managed(f64),
};
pub const Metrics = struct {
    mse: f64,
    rmse: f64,
};

const std = @import("std");
