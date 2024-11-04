const std = @import("std");
const csv = @import("csv.zig");

//struct declartion for the dataset

const DataSet = struct { x: []f64, y: []f64 };

const Derivative = struct { dm: f64, db: f64 };

const Regression = struct { m: f64, b: f64 };

const HyperParameters = struct {
    epochs: usize,
    learning_rate: f64,
    name: []const u8,
};

const EvaluationResult = struct {
    predictions: std.ArrayList(f64),
};

const Metrics = struct {
    mse: f64,
    rmse: f64,
};

pub fn evaluate_model(regression: Regression, dataset: DataSet) !EvaluationResult {
    var predictions = std.ArrayList(f64).init(std.heap.page_allocator);

    // Evaluate on dataset
    for (dataset.x) |x| {
        const pred = predict(regression, x);
        try predictions.append(pred);
    }

    return EvaluationResult{
        .predictions = predictions,
    };
}

pub fn calculate_metrics(predictions: []const f64, dataset: DataSet) Metrics {
    var total_error: f64 = 0.0;

    for (predictions, dataset.y) |pred, y| {
        const test_error = pred - y;
        total_error += test_error * test_error;
    }

    const mse = total_error / @as(f64, @floatFromInt(dataset.y.len));
    const rmse = @sqrt(mse);

    return Metrics{
        .mse = mse,
        .rmse = rmse,
    };
}

pub fn run_experiment(train_dataset: DataSet, test_dataset: DataSet, hyperparams: HyperParameters) !void {
    const train_start: i64 = std.time.milliTimestamp();
    const regression = train(train_dataset, hyperparams);
    const train_end: i64 = std.time.milliTimestamp();

    const train_time = train_end - train_start;

    std.debug.print("Training completed in {d}ms\n", .{train_time});
    std.debug.print("Regression model: m = {d}, b = {d}\n", .{ regression.m, regression.b });

    const evaluation_result = try evaluate_model(regression, test_dataset);
    defer evaluation_result.predictions.deinit();

    const metrics = calculate_metrics(evaluation_result.predictions.items, test_dataset);

    std.debug.print("\nModel Performance Metrics for {s}:\n", .{hyperparams.name});
    std.debug.print("Mean Squared Error (MSE): {d:.6}\n", .{metrics.mse});
    std.debug.print("Root Mean Squared Error (RMSE): {d:.6}\n", .{metrics.rmse});
}

pub fn main() !void {
    std.debug.print("Starting Regression Experiments!\n", .{});

    const filename = "data/data_for_lr.csv";
    const content = try csv.read_csv(filename);
    defer std.heap.page_allocator.free(content);

    const dataset = try parse_csv_data(content);
    defer std.heap.page_allocator.free(dataset.x);
    defer std.heap.page_allocator.free(dataset.y);

    const train_dataset = DataSet{ .x = dataset.x[0..500], .y = dataset.y[0..500] };
    const test_dataset = DataSet{ .x = dataset.x[500..], .y = dataset.y[500..] };

    // Define different hyperparameter configurations
    const experiments = [_]HyperParameters{
        .{ .epochs = 5000, .learning_rate = 0.00001, .name = "Fast Learning" },
        .{ .epochs = 10000, .learning_rate = 0.000005, .name = "Medium Learning" },
        .{ .epochs = 20000, .learning_rate = 0.000001, .name = "Slow Learning" },
    };

    // Run each experiment
    for (experiments) |hyperparams| {
        try run_experiment(train_dataset, test_dataset, hyperparams);
    }
}

pub fn normalize(data: []f64) void {
    const mean_val = mean(data);
    const std_dev = standard_deviation(data);
    for (data) |*value| {
        value.* = (value.* - mean_val) / std_dev;
    }
}

pub fn train(dataset: DataSet, hyperparams: HyperParameters) Regression {
    const length = dataset.y.len;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    // Initialize m and b with small random values between -1 and 1
    var regression = Regression{
        .m = rand.float(f64) * 2.0 - 1.0, // Random value between -1 and 1
        .b = rand.float(f64) * 2.0 - 1.0, // Random value between -1 and 1
    };

    std.debug.print("\nStarting training with hyperparameters:\n", .{});
    std.debug.print("Experiment: {s}\n", .{hyperparams.name});
    std.debug.print("Epochs: {d}\n", .{hyperparams.epochs});
    std.debug.print("Learning rate: {d}\n", .{hyperparams.learning_rate});
    normalize(dataset.x);
    normalize(dataset.y);

    var counter: usize = 0;
    while (counter <= hyperparams.epochs) {
        var predictions = std.heap.page_allocator.alloc(f64, length) catch unreachable;
        defer std.heap.page_allocator.free(predictions);
        var i: usize = 0;
        for (dataset.x) |x| {
            const prediction = predict(regression, x);
            predictions[i] = prediction;
            i += 1;
        }

        const derivative = backwards_prop(predictions, dataset);
        regression.m -= hyperparams.learning_rate * derivative.dm;
        regression.b -= hyperparams.learning_rate * derivative.db;

        const run_cost = cost(predictions, dataset);
        if (counter % 1000 == 0) {
            std.debug.print("Epoch: {d}, Cost: {d}\n", .{ counter, run_cost });
        }
        counter += 1;
    }
    return regression;
}

pub fn predict(regression: Regression, x: f64) f64 {
    return regression.m * x + regression.b;
}

pub fn backwards_prop(predictions: []f64, dataset: DataSet) Derivative {
    const length = dataset.y.len;
    const n = @as(f64, @floatFromInt(length));
    var i: usize = 0;
    //Calculate DF
    //
    var df = std.heap.page_allocator.alloc(f64, length) catch unreachable;
    defer std.heap.page_allocator.free(df);
    for (predictions, dataset.y) |prediction, y| {
        df[i] = prediction - y;
        i += 1;
    }

    i = 0;
    //Calculate dm
    var sum_dm: f64 = 0.0;
    for (df, dataset.x) |df_val, x| {
        sum_dm += (df_val * x);
        i += 1;
    }
    const dm = 2 * sum_dm / n;

    i = 0;
    //Calculate db
    var sum_db: f64 = 0.0;
    for (df) |df_val| {
        sum_db += df_val;
        i += 1;
    }
    const db = 2 * sum_db / n;

    return Derivative{ .dm = dm, .db = db };
}

pub fn cost(predictions: []f64, dataset: DataSet) f64 {
    const n = @as(f64, @floatFromInt(dataset.y.len));
    var sum: f64 = 0.0;
    for (predictions, dataset.y) |prediction, y| {
        sum += (prediction - y) * (prediction - y);
    }
    return sum / n;
}

//pub fn train(dataset: DataSet) ![]f64 {}

pub fn mean(data: []f64) f64 {
    var sum: f64 = 0.0;
    const n = @as(f64, @floatFromInt(data.len));
    for (data) |value| {
        sum += value;
    }
    return sum / n;
}

pub fn standard_deviation(data: []f64) f64 {
    std.debug.print("Calculating standard deviation\n", .{});
    const n = @as(f64, @floatFromInt(data.len));
    const data_mean = mean(data);
    var sum: f64 = 0.0;
    for (data) |value| {
        sum += @exp2((value - data_mean) / n);
    }
    return @sqrt(sum);
}

fn parse_csv_data(content: []u8) !DataSet {
    const allocator = std.heap.page_allocator;
    var x_list = std.ArrayList(f64).init(allocator);
    var y_list = std.ArrayList(f64).init(allocator);
    defer x_list.deinit();
    defer y_list.deinit();

    var iterator = std.mem.split(u8, content, "\n");
    var line_number: usize = 0;

    while (iterator.next()) |line| {
        line_number += 1;
        if (line.len == 0) continue;
        if (line[0] == 'x') continue;

        var data = std.mem.split(u8, line, ",");

        const x_str = data.next() orelse continue;
        const trimmed_x = std.mem.trim(u8, x_str, &std.ascii.whitespace);
        const x = std.fmt.parseFloat(f64, trimmed_x) catch |err| {
            std.debug.print("Error parsing x at line {d}: {s} - {}\n", .{ line_number, x_str, err });
            continue;
        };

        const y_str = data.next() orelse continue;
        // Trim the y value
        const trimmed_y = std.mem.trim(u8, y_str, &std.ascii.whitespace);
        const y = std.fmt.parseFloat(f64, trimmed_y) catch |err| {
            std.debug.print("Error parsing y at line {d}: '{s}' - {}\n", .{ line_number, trimmed_y, err });
            continue;
        };

        try x_list.append(x);
        try y_list.append(y);
    }
    //

    return DataSet{ .x = try x_list.toOwnedSlice(), .y = try y_list.toOwnedSlice() };
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
