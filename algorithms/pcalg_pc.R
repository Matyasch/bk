library(pcalg)


methods::setRefClass("TestCounter",
    fields = c("tests_done", "test_num", "indepTest"),
    methods = list(
        initialize = function(indepTest, node_num) {
            tests_done <<- lapply(
                seq_len(node_num),
                function(.) vector("list", node_num)
            )
            test_num <<- numeric(node_num)
            indepTest <<- indepTest
        },
        test = function(x, y, S = NULL, suffStat) {
            # Make sure that x, y, and S are the same format.
            x <- unname(x)
            y <- unname(y)
            S <- unname(S)
            # Order variables.
            if (x > y) {
                x_ <- x
                x <- y
                y <- x_
            }
            # S <- sort(S)
            # Keep track of test.
            old_length <- length(tests_done[[x]][[y]])
            tests_done[[x]][[y]] <<- union(tests_done[[x]][[y]], list(S))
            if (length(tests_done[[x]][[y]]) > old_length) {
                test_num[length(S) + 1] <<- test_num[length(S) + 1] + 1
            }
            # Run the test.
            return(indepTest(x, y, S, suffStat))
        }
    )
)


pcalg_pc <- function(data, ci_test, alpha, bk, skel_method, dag) {
    # CI test
    dag <- as(dag, "graphNEL")
    samples_num <- nrow(data)
    cor <- cor(data)
    suffStat <- list(
        g = dag, jp = RBGL::johnson.all.pairs.sp(dag), # pcalg::dsepTest
        C = cor, n = samples_num, # pcalg::gaussCItest
        dm = data, adaptDF = TRUE # pcalg::disCItest
    )
    indepTest <- switch(ci_test,
        d_separation = pcalg::dsepTest,
        fisherz = pcalg::gaussCItest,
        gsq = pcalg::disCItest,
        stop(paste0("Unknown ci_test: '", ci_test, "'"))
    )
    counter <- methods::getRefClass("TestCounter")(indepTest, ncol(data))

    # BK
    fixedEdges <- matrix(as.numeric(bk == -1 | t(bk) == -1), nrow = nrow(bk), ncol = ncol(bk))
    fixedGaps <- matrix(as.numeric(bk == 1 & t(bk) == 1), nrow = nrow(bk), ncol = ncol(bk))

    # Run PC algorithm
    pc_res <- pcalg::pc(
        suffStat = suffStat,
        indepTest = counter$test,
        alpha = alpha,
        labels = suffStat$g@nodes,
        skel.method = skel_method,
        fixedEdges = fixedEdges,
        fixedGaps = fixedGaps
    )

    cpdag <- as(pc_res, "amat")
    # Directed BK: bk[i,j] == -1 & bk[j,i] == 1  =>  orient edge i -> j
    dir_idx <- which(bk == -1 & t(bk) == 1, arr.ind = TRUE)
    if (nrow(dir_idx) > 0) {
        mpdag <- pcalg::addBgKnowledge(
            gInput = cpdag,
            x = dir_idx[, "row"],
            y = dir_idx[, "col"],
            checkInput = FALSE,
            verbose = FALSE
        )
    } else {
        mpdag <- NULL
    }

    if(is.null(mpdag)) {
        mpdag <- cpdag
    }

    # Convert results
    out <- list(
        amat = mpdag,
        test_num = counter$test_num
    )
    return(out)
}