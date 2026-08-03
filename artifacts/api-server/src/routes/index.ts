import { Router, type IRouter } from "express";
import healthRouter from "./health";
import authRouter from "./auth";
import personnelRouter from "./personnel";
import accountsRouter from "./accounts";
import logsRouter from "./logs";
import statsRouter from "./stats";

const router: IRouter = Router();

router.use(healthRouter);
router.use("/auth", authRouter);
router.use("/personnel", personnelRouter);
router.use("/accounts", accountsRouter);
router.use("/logs", logsRouter);
router.use("/stats", statsRouter);

export default router;
