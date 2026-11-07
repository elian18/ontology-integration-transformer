import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.apache.jena.rdf.model.Literal;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.sys.JenaSystem;

// Usage: java -cp "<jena>/lib/*" QuestionRunner.java QUERY.rq... ++ NAME FILE... [-- NAME FILE...]
// Loads each set of files into one model (no inference) and runs every query on it.
// Prints one JSON object: the rows of each query in each set.
public class QuestionRunner {

    public static void main(String[] args) throws Exception {
        JenaSystem.init();
        List<String> queries = new ArrayList<>();
        int i = 0;
        while (i < args.length && !args[i].equals("++")) {
            queries.add(args[i++]);
        }
        i++;
        List<String> sets = new ArrayList<>();
        while (i < args.length) {
            String name = args[i++];
            List<String> files = new ArrayList<>();
            while (i < args.length && !args[i].equals("--")) {
                files.add(args[i++]);
            }
            i++;
            sets.add(quote(name) + ":" + run(files, queries));
        }
        System.out.println("{\"sets\":{" + String.join(",", sets) + "}}");
    }

    static String run(List<String> files, List<String> queries) throws Exception {
        long start = System.nanoTime();
        Model data = ModelFactory.createDefaultModel();
        for (String file : files) {
            RDFDataMgr.read(data, file);
        }
        List<String> answers = new ArrayList<>();
        for (String file : queries) {
            Path path = Path.of(file);
            String id = path.getFileName().toString().replaceFirst("\\.rq$", "");
            answers.add(quote(id) + ":" + answer(Files.readString(path, StandardCharsets.UTF_8), data));
        }
        long millis = (System.nanoTime() - start) / 1_000_000;
        return "{\"triples\":" + data.size() + ",\"millis\":" + millis + ",\"answers\":{"
                + String.join(",", answers) + "}}";
    }

    static String answer(String text, Model data) {
        try {
            Query query = QueryFactory.create(text);
            try (QueryExecution exec = QueryExecutionFactory.create(query, data)) {
                ResultSet results = exec.execSelect();
                List<String> vars = results.getResultVars();
                List<String> rows = new ArrayList<>();
                while (results.hasNext()) {
                    QuerySolution row = results.next();
                    List<String> cells = new ArrayList<>();
                    for (String v : vars) {
                        cells.add(cell(row.get(v)));
                    }
                    rows.add("[" + String.join(",", cells) + "]");
                }
                List<String> names = new ArrayList<>();
                for (String v : vars) {
                    names.add(quote(v));
                }
                return "{\"vars\":[" + String.join(",", names) + "],\"rows\":["
                        + String.join(",", rows) + "]}";
            }
        } catch (Exception e) {
            return "{\"error\":" + quote(e.getClass().getSimpleName() + ": " + e.getMessage()) + "}";
        }
    }

    static String cell(RDFNode node) {
        if (node == null) {
            return "null";
        }
        if (node.isAnon()) {
            return "{\"t\":\"bnode\"}";
        }
        if (node.isResource()) {
            return "{\"t\":\"uri\",\"v\":" + quote(node.asResource().getURI()) + "}";
        }
        Literal literal = node.asLiteral();
        String lang = literal.getLanguage();
        return "{\"t\":\"literal\",\"v\":" + quote(literal.getLexicalForm()) + ",\"dt\":"
                + quote(literal.getDatatypeURI()) + ",\"lang\":"
                + quote(lang == null || lang.isEmpty() ? null : lang) + "}";
    }

    static String quote(String text) {
        if (text == null) {
            return "null";
        }
        StringBuilder out = new StringBuilder("\"");
        for (char c : text.toCharArray()) {
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20 || c > 0x7e) {          // ASCII only: any console encoding
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.append('"').toString();
    }
}
