const std = @import("std");

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
