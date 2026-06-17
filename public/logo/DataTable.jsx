import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table"

export default function DataTable() {
  if (!props.rows || props.rows.length === 0) {
    return <div>No data available</div>
  }

  const headers = Object.keys(props.rows[0])

  return (
    <Card className="w-full overflow-x-auto">
      <CardHeader>
        <CardTitle>{props.title || "Data Table"}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              {headers.map((h) => (
                <TableHead key={h}>{h}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {props.rows.map((row, idx) => (
              <TableRow key={idx}>
                {headers.map((h) => (
                  <TableCell key={h}>{String(row[h])}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
