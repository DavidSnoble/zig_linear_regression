const std = @import("std");

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

pub fn normalize(data: []f64) void {
    const mean_val = mean(data);
    const std_dev = standard_deviation(data);
    for (data) |*value| {
        value.* = (value.* - mean_val) / std_dev;
    }
}
