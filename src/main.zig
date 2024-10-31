const std = @import("std");

pub fn main() !void {
    const filename = "data/data_for_lr.csv";

    std.debug.print("Opening CSV {s}\n", .{filename});

    // initialize a head allocator
    const allocator = std.heap.page_allocator;
    // open a file
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    var buffer_reader = std.io.bufferedReader(file.reader());
    const reader = buffer_reader.reader();

    var line = std.ArrayList(u8).init(allocator);
    defer line.deinit();

    const writer = line.writer();
    var line_no: usize = 0;
    while (reader.streamUntilDelimiter(writer, '\n', null)) {
        // Clear line so we can reuse it.
        defer line.clearRetainingCapacity();
        line_no += 1;

        std.debug.print("{d}--{s}\n", .{ line_no, line.items });
    } else |err| switch (err) {
        error.EndOfStream => {
            if (line.items.len > 0) {
                line_no += 1;
                std.debug.print("{d}--{s}\n", .{ line_no, line.items });
            }
        },
        else => return err,
    }

    std.debug.print("Total lines: {d}\n", .{line_no});
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
