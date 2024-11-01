const std = @import("std");

//struct declartion for the dataset

const DataSet = struct { x: []f64, y: []f64 };

const Derivative = struct { dm: f64, db: f64 };

const Regression = struct { m: f64, b: f64 };

pub fn main() !void {
    std.debug.print("Starting Regression!\n", .{});

    const filename = "data/data_for_lr.csv";
    const content = try read_csv(filename);
    defer std.heap.page_allocator.free(content);

    // parse csv into x and y vectors
    const dataset = try parse_csv_data(content);
    defer std.heap.page_allocator.free(dataset.x);
    defer std.heap.page_allocator.free(dataset.y);

    std.debug.print("Dataset loaded", .{});

    const train_dataset = DataSet{ .x = dataset.x[0..500], .y = dataset.y[0..500] };
    const test_dataset = DataSet{ .x = dataset.x[500..], .y = dataset.y[500..] };

    std.debug.print("size of train_dataset: {d}\n", .{train_dataset.y.len});
    std.debug.print("size of test_dataset: {d}\n", .{test_dataset.y.len});

    const x_mean = mean(dataset.x);
    const y_mean = mean(dataset.y);

    std.debug.print("Mean of x: {d}\n", .{x_mean});
    std.debug.print("Mean of y: {d}\n", .{y_mean});

    const regresssion: Regression = train(train_dataset);
    std.debug.print("Regression model: m = {d}, b = {d}\n", .{ regresssion.m, regresssion.b });
    //for (dataset.x, dataset.y) |x, y| {
    //    std.debug.print("{d}, {d}\n", .{ x, y });
    //}
}

pub fn normalize(data: []f64) void {
    const mean_val = mean(data);
    const std_dev = standard_deviation(data);
    for (data) |*value| {
        value.* = (value.* - mean_val) / std_dev;
    }
}

pub fn train(dataset: DataSet) Regression {
    const length = dataset.y.len;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    // Initialize m and b with small random values between -1 and 1
    var regression = Regression{
        .m = rand.float(f64) * 2.0 - 1.0, // Random value between -1 and 1
        .b = rand.float(f64) * 2.0 - 1.0, // Random value between -1 and 1
    };
    //
    const epochs = 200000;
    const learning_rate = 0.00001;
    normalize(dataset.x);
    normalize(dataset.y);

    var counter: usize = 0;
    while (counter <= epochs) {
        var predictions = std.heap.page_allocator.alloc(f64, length) catch unreachable;
        defer std.heap.page_allocator.free(predictions);
        var i: usize = 0;
        for (dataset.x) |x| {
            const prediction = predict(regression, x);
            predictions[i] = prediction;
            i += 1;
        }

        const derivative = backwards_prop(predictions, dataset);
        regression.m -= learning_rate * derivative.dm;
        regression.b -= learning_rate * derivative.db;

        const run_cost = cost(predictions, dataset);

        std.debug.print("Epoch: {d}, Cost: {d}\n", .{ counter, run_cost });
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

// Read a CSV file and return the content as a byte array
pub fn read_csv(filename: []const u8) ![]u8 {
    std.debug.print("Opening CSV {s}\n", .{filename});

    // initialize a heap allocator
    const allocator = std.heap.page_allocator;
    // open a file
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    var buffer_reader = std.io.bufferedReader(file.reader());
    const reader = buffer_reader.reader();

    var line = std.ArrayList(u8).init(allocator);
    defer line.deinit();

    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();

    const writer = line.writer();
    var line_no: usize = 0;
    while (reader.streamUntilDelimiter(writer, '\n', null)) {
        line_no += 1;
        try buffer.appendSlice(line.items);
        try buffer.append('\n');
        line.clearRetainingCapacity();
    } else |err| switch (err) {
        error.EndOfStream => {
            if (line.items.len > 0) {
                line_no += 1;
                try buffer.appendSlice(line.items);
                try buffer.append('\n');
            }
        },
        else => return err,
    }

    std.debug.print("Total lines: {d}\n", .{line_no});
    return buffer.toOwnedSlice();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
