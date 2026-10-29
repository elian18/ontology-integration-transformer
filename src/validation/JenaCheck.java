import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.reasoner.ValidityReport;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.sys.JenaSystem;

// Usage: java -cp "<jena>/lib/*" JenaCheck.java REASONER NAME FILE... [-- NAME FILE...]
// Prints one JSON object: the validity report of each set of files.
public class JenaCheck {

    public static void main(String[] args) {
        JenaSystem.init();
        Reasoner reasoner = switch (args[0]) {
            case "rdfs" -> ReasonerRegistry.getRDFSReasoner();
            case "owl-micro" -> ReasonerRegistry.getOWLMicroReasoner();
            case "owl-mini" -> ReasonerRegistry.getOWLMiniReasoner();
            case "owl" -> ReasonerRegistry.getOWLReasoner();
            default -> throw new IllegalArgumentException("Unknown reasoner: " + args[0]);
        };
        List<String> sets = new ArrayList<>();
        int i = 1;
        while (i < args.length) {
            String name = args[i++];
            List<String> files = new ArrayList<>();
            while (i < args.length && !args[i].equals("--")) {
                files.add(args[i++]);
            }
            i++;
            sets.add(quote(name) + ":" + check(files, reasoner));
        }
        System.out.println("{\"reasoner\":" + quote(args[0]) + ",\"sets\":{"
                + String.join(",", sets) + "}}");
    }

    static String check(List<String> files, Reasoner reasoner) {
        long start = System.nanoTime();
        Model data = ModelFactory.createDefaultModel();
        for (String file : files) {
            RDFDataMgr.read(data, file);
        }
        InfModel inferred = ModelFactory.createInfModel(reasoner, data);
        ValidityReport validity = inferred.validate();
        List<String> reports = new ArrayList<>();
        for (Iterator<ValidityReport.Report> it = validity.getReports(); it.hasNext(); ) {
            ValidityReport.Report r = it.next();
            reports.add("{\"error\":" + r.isError() + ",\"type\":" + quote(r.getType())
                    + ",\"description\":" + quote(r.getDescription())
                    + ",\"culprit\":" + quote(String.valueOf(r.getExtension())) + "}");
        }
        long millis = (System.nanoTime() - start) / 1_000_000;
        return "{\"triples\":" + data.size() + ",\"valid\":" + validity.isValid()
                + ",\"clean\":" + validity.isClean() + ",\"millis\":" + millis
                + ",\"reports\":[" + String.join(",", reports) + "]}";
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